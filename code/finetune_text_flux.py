import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import FluxPipeline
from tqdm.auto import tqdm
from PIL import Image

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model_id = "ostris/OpenFLUX.1"
pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipeline = pipeline.to(device)

# Custom dataset class with text labels
class CustomImageTextDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = [f.split('_')[0].replace('-', ' ') for f in self.images]  # Assuming filenames are in format: "label_image.jpg"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Set up data
transform = transforms.Compose([
    transforms.Resize(1024, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = CustomImageTextDataset("path/to/your/image/directory", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training function
def train_model(pipeline, dataloader, optimizer, num_epochs, device):
    pipeline.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, labels in tqdm(dataloader):
            clean_images = batch.to(device)
            
            # Encode text
            (prompt_embeds, pooled_prompt_embeds, text_ids) = pipeline.encode_prompt(prompt=labels, device=device)
            
            # Encode images
            latents = pipeline.vae.encode(clean_images).latent_dist.sample() * pipeline.vae.config.scaling_factor
            
            # Add noise to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = pipeline.scheduler.scale_noise(latents, timesteps, noise)
            
            # Predict the noise residual
            noise_pred = pipeline.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Training
print("Training FLUX...")
optimizer = torch.optim.AdamW(pipeline.transformer.parameters(), lr=1e-5)
train_model(pipeline, dataloader, optimizer, num_epochs=100, device=device)

# Save fine-tuned model
pipeline.save_pretrained("path/to/save/fine_tuned_model")

print("Fine-tuning complete!")

# Example of generating an image with the fine-tuned model
prompt = "a photo of a cat"
image = pipeline(prompt).images[0]
image.save("generated_image.png")
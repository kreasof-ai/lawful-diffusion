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

# Custom dataset class (remains the same)
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Set up data
transform = transforms.Compose([
    transforms.Resize(1024, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = CustomImageDataset("path/to/your/image/directory", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training function
def train_model(model, dataloader, optimizer, num_epochs, device, is_vae=False):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            clean_images = batch.to(device)
            
            if is_vae:
                # VAE training
                encoder_output = model.encode(clean_images) * pipeline.vae.config.scaling_factor
                z = encoder_output.latent_dist.sample()
                reconstructed_images = model.decode(z / pipeline.vae.config.scaling_factor).sample
                
                loss = torch.nn.functional.mse_loss(reconstructed_images, clean_images)
            else:
                # Transformer training
                latents = pipeline.vae.encode(clean_images).latent_dist.sample() * pipeline.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = pipeline.scheduler.scale_noise(latents, timesteps, noise)
                noise_pred = model(hidden_states=noisy_latents, timestep=timesteps / 1000, return_dict=False)[0]
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# VAE Training
print("Training VAE...")
vae_optimizer = torch.optim.AdamW(pipeline.vae.parameters(), lr=1e-5)
train_model(pipeline.vae, dataloader, vae_optimizer, num_epochs=3, device=device, is_vae=True)

# Save fine-tuned VAE
pipeline.vae.save_pretrained("path/to/save/fine_tuned_vae")

# Transformers Training
print("Training Transformer...")
transformer_optimizer = torch.optim.AdamW(pipeline.transformer.parameters(), lr=1e-4)
train_model(pipeline.transformer, dataloader, transformer_optimizer, num_epochs=10, device=device, is_vae=False)

# Save fine-tuned UNet
pipeline.transformer.save_pretrained("path/to/save/fine_tuned_transformer")

print("Fine-tuning complete!")

# Optional: Combine and save the full pipeline
pipeline.save_pretrained("path/to/save/fine_tuned_model")
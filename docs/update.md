To update your implementation of Lawful Diffusion by fine-tuning both the generative text-to-image model (Stable Diffusion) and the classifier head, you'll need to extend your current setup to include the fine-tuning process for Stable Diffusion alongside the classifier training. Here's a comprehensive guide and updated code to achieve this.

### **Key Steps to Implement Fine-Tuning for Both Models:**

1. **Prepare the Dataset:**
   - Ensure your dataset includes paired text prompts and images. If not, you might need to generate or associate text prompts with images.
   
2. **Modify the Dataset Class:**
   - Adjust the `ArtistDataset` to return both text prompts and images.

3. **Set Up Optimizers:**
   - Define separate optimizers for the Stable Diffusion components and the classifier head.

4. **Define Training Loops:**
   - Implement simultaneous training for both the generative model and the classifier.
   - Compute separate loss functions for image generation and classification, then combine them.

5. **Manage Computational Resources:**
   - Fine-tuning Stable Diffusion is resource-intensive. Ensure you have adequate GPU memory.
   - Consider using techniques like **gradient checkpointing** or **mixed-precision training** to optimize resource usage.

6. **Save and Upload Both Models:**
   - After training, save both the fine-tuned Stable Diffusion model and the classifier head.
   - Upload them to Hugging Face Hub as needed.

### **Updated Code Implementation:**

Below is the updated code incorporating the fine-tuning of both the Stable Diffusion model and the classifier head. Please read through the comments for clarity.

```python
import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIPTokenizer
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from datasets import load_dataset
from safetensors.torch import save_file
from huggingface_hub import HfApi, upload_folder
from torch.cuda.amp import GradScaler, autocast

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Pretrained Models
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# Move components to device
sd_pipeline.to(device)

# Extract components for fine-tuning
vae = sd_pipeline.vae
text_encoder = sd_pipeline.text_encoder
tokenizer = sd_pipeline.tokenizer
unet = sd_pipeline.unet

# Set to train mode
vae.train()
text_encoder.train()
unet.train()

# Load CLIP model for classifier
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)
clip_model.eval()

# 2. Define Transformation for VAE
vae_transform = Compose([
    Resize(512, interpolation=Image.BICUBIC),
    CenterCrop(512),
    ToTensor(),
    Normalize([0.5], [0.5])
])

# 3. Define Embedding Extraction Functions
def get_clip_embedding(image, clip_processor, clip_model, device):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**inputs)
    clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)
    return clip_features.cpu()

def get_vae_embedding(image, vae, vae_transform, device):
    image = vae_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        vae_features = latent_dist.mean
    return vae_features.cpu()

def get_combined_embedding(image, clip_processor, clip_model, vae, vae_transform, device):
    clip_emb = get_clip_embedding(image, clip_processor, clip_model, device)
    vae_emb = get_vae_embedding(image, vae, vae_transform, device)
    combined_emb = torch.cat([clip_emb, vae_emb], dim=1)
    return combined_emb

# 4. Define Custom Dataset using Hugging Face datasets
class ArtistDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, clip_processor, clip_model, vae, vae_transform, label_encoder, max_length=77):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vae = vae
        self.vae_transform = vae_transform
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        artist = item['artist']
        prompt = item.get('text', f"Artwork by {artist}")  # Assuming 'text' field exists; else use default.

        # Tokenize prompt
        inputs = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = inputs.input_ids.squeeze(0)  # Shape: (max_length,)
        attention_mask = inputs.attention_mask.squeeze(0)

        # Get combined embedding for classifier
        combined_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vae, self.vae_transform, device)

        # Get label
        label = self.label_encoder.transform([artist])[0]

        # Process image for diffusion model
        image = self.vae_transform(image)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': image,
            'labels': image,  # For diffusion models, labels are typically the same as pixel_values
            'combined_emb': combined_emb.squeeze(0),
            'label': label
        }

# 5. Initialize Label Encoder and Save
dataset_name = "your_username/your_dataset_name"  # Replace with your Hugging Face dataset name
dataset = load_dataset(dataset_name, split='train')

# Gather artist names
artist_names = dataset.unique('artist')

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(artist_names)

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 6. Create Dataset and DataLoader
batch_size = 4  # Reduced batch size due to high memory consumption
num_workers = 4

train_dataset = ArtistDataset(
    dataset_name=dataset_name,
    split='train',
    tokenizer=tokenizer,
    clip_processor=clip_processor,
    clip_model=clip_model,
    vae=vae,
    vae_transform=vae_transform,
    label_encoder=label_encoder
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# 7. Define the Classification Model
class ArtistClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ArtistClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

# Determine input dimensions
clip_dim = clip_model.config.projection_dim if hasattr(clip_model.config, 'projection_dim') else 512
vae_dim = vae.config.latent_channels * vae.config.block_out_channels[-1]  # Adjust based on VAE architecture
input_dim = clip_dim + vae_dim

num_classes = len(artist_names)

model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# 8. Define Loss and Optimizer
criterion_cls = nn.CrossEntropyLoss()
criterion_unet = nn.MSELoss()  # Typical for diffusion models

# Optimizer for classifier
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4)

# Optimizer for Stable Diffusion components (fine-tuning UNet and Text Encoder)
optimizer_sd = optim.Adam(
    list(unet.parameters()) + list(text_encoder.parameters()),
    lr=1e-5
)

# 9. Mixed Precision and Gradient Scaler
scaler = GradScaler()

# 10. Training Loop
num_epochs = 3  # Adjust based on your needs and resources

for epoch in range(num_epochs):
    model.train()
    unet.train()
    text_encoder.train()
    total_loss_cls = 0
    total_loss_unet = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        combined_emb = batch['combined_emb'].to(device)
        label = batch['label'].to(device)

        # Zero the gradients
        optimizer_cls.zero_grad()
        optimizer_sd.zero_grad()

        with autocast():
            # === Fine-Tuning Stable Diffusion ===
            # Encode the text
            encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

            # Sample noise and add to images (for diffusion)
            noise = torch.randn_like(pixel_values)
            timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=device).long()
            noisy_images = noise.add_0(pixel_values)  # Placeholder for actual noise addition based on timesteps

            # Get model prediction
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample

            # Compute diffusion loss
            loss_unet = criterion_unet(noise_pred, noise)

            # === Fine-Tuning Classifier ===
            # Forward pass through classifier
            logits = model(combined_emb)
            loss_cls = criterion_cls(logits, label)

            # === Combine Losses ===
            # Weight the losses as needed
            total_loss = loss_unet + loss_cls

        # Backpropagation with mixed precision
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_sd)
        scaler.step(optimizer_cls)
        scaler.update()

        total_loss_unet += loss_unet.item()
        total_loss_cls += loss_cls.item()

    avg_loss_unet = total_loss_unet / len(train_dataloader)
    avg_loss_cls = total_loss_cls / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss UNet: {avg_loss_unet:.4f}, Loss Classifier: {avg_loss_cls:.4f}")

# 11. Save the Trained Models using safetensors and diffsuers' save_pretrained
# Save the classification head
save_file(model.state_dict(), "artist_classifier.safetensors")
print("Classifier saved to artist_classifier.safetensors")

# Save the fine-tuned Stable Diffusion model components
sd_output_dir = "fine_tuned_stable_diffusion"
os.makedirs(sd_output_dir, exist_ok=True)

# Save UNet
unet.save_pretrained(os.path.join(sd_output_dir, "unet"))

# Save Text Encoder
text_encoder.save_pretrained(os.path.join(sd_output_dir, "text_encoder"))

# Save VAE (if needed)
vae.save_pretrained(os.path.join(sd_output_dir, "vae"))

print(f"Stable Diffusion components saved to {sd_output_dir}")

# 12. Upload the Models to Hugging Face Hub
api = HfApi()
repo_id_classifier = "your-username/artist-classifier"  # Replace with your repository name for classifier
repo_id_sd = "your-username/fine-tuned-stable-diffusion"  # Replace with your repository name for SD

# Create repositories if they don't exist
api.create_repo(repo_id_classifier, exist_ok=True)
api.create_repo(repo_id_sd, exist_ok=True)

# Upload the classifier
api.upload_file(
    path_or_fileobj="artist_classifier.safetensors",
    path_in_repo="artist_classifier.safetensors",
    repo_id=repo_id_classifier,
)

# Upload the label encoder
api.upload_file(
    path_or_fileobj="label_encoder.pkl",
    path_in_repo="label_encoder.pkl",
    repo_id=repo_id_classifier,
)

# Upload the fine-tuned Stable Diffusion components
api.upload_folder(
    folder_path=sd_output_dir,
    repo_id=repo_id_sd,
    repo_type="model"
)

print(f"Models uploaded to {repo_id_classifier} and {repo_id_sd}")

# 13. Inference Functions
def generate_image_with_artist_reference(prompt, sd_pipeline, model, clip_processor, clip_model, vae, vae_transform, label_encoder, device, tokenizer, top_k=3):
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate Image
    with torch.no_grad():
        with torch.autocast(device.type):
            generated_image = sd_pipeline(prompt).images[0]

    # Get combined embedding
    combined_emb = get_combined_embedding(generated_image, clip_processor, clip_model, vae, vae_transform, device)
    combined_emb_tensor = combined_emb.to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(combined_emb_tensor.unsqueeze(0))
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_labels = torch.topk(probabilities, top_k)

    # Decode labels
    predicted_artists = label_encoder.inverse_transform(top_labels.cpu().numpy().flatten())

    # Prepare attribution
    attribution = []
    for i in range(top_k):
        attribution.append({
            "artist": predicted_artists[i],
            "probability": top_probs[0][i].item()
        })

    return generated_image, attribution

def verify_external_image_enhanced(image_path, model, clip_processor, clip_model, vae, vae_transform, device, label_encoder, top_k=5):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Get combined embedding
    combined_emb = get_combined_embedding(image, clip_processor, clip_model, vae, vae_transform, device)
    combined_emb_tensor = combined_emb.to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(combined_emb_tensor.unsqueeze(0))
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_labels = torch.topk(probabilities, top_k)

    # Decode labels
    predicted_artists = label_encoder.inverse_transform(top_labels.cpu().numpy().flatten())

    # Prepare verification report
    verification_report = []
    for i in range(top_k):
        verification_report.append({
            "artist": predicted_artists[i],
            "probability": top_probs[0][i].item()
        })

    return verification_report

# 14. Example Usage
if __name__ == "__main__":
    # Reload the models (if needed)
    # Load classifier
    model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("artist_classifier.safetensors"))
    model.eval()

    # Load fine-tuned Stable Diffusion
    sd_finetuned_pipeline = StableDiffusionPipeline.from_pretrained(
        sd_output_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    sd_finetuned_pipeline.to(device)

    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Example Prompt
    prompt = "A futuristic cityscape at sunset"
    generated_image, attribution = generate_image_with_artist_reference(
        prompt=prompt,
        sd_pipeline=sd_finetuned_pipeline,
        model=model,
        clip_processor=clip_processor,
        clip_model=clip_model,
        vae=vae,
        vae_transform=vae_transform,
        label_encoder=label_encoder,
        device=device,
        tokenizer=tokenizer,
        top_k=3
    )

    # Display the generated image
    generated_image.show()

    # Print attribution
    print("\nAttribution:")
    for idx, attrib in enumerate(attribution, 1):
        print(f"{idx}. Artist: {attrib['artist']}, Probability: {attrib['probability']:.2f}")

    # Example External Image Verification
    external_image_path = "path_to_external_image.jpg"  # Replace with your image path
    verification_report = verify_external_image_enhanced(
        image_path=external_image_path,
        model=model,
        clip_processor=clip_processor,
        clip_model=clip_model,
        vae=vae,
        vae_transform=vae_transform,
        device=device,
        label_encoder=label_encoder,
        top_k=5
    )

    if verification_report:
        print("\nVerification Report:")
        for idx, report in enumerate(verification_report, 1):
            print(f"{idx}. Artist: {report['artist']}, Probability: {report['probability']:.2f}")
    else:
        print("\nNo similar images found or error in processing.")
```

### **Detailed Explanation of the Updates:**

1. **Dataset Preparation:**
   - **Text Prompts:** The `ArtistDataset` class now includes text prompts. If your dataset doesn't have a 'text' field, it defaults to "Artwork by [artist]". Ensure your dataset includes meaningful prompts for better fine-tuning.
   - **Data Returned:** The dataset returns `input_ids`, `attention_mask` for the text encoder, `pixel_values` for the image, `labels` for diffusion training, and `combined_emb` along with the classification `label`.

2. **Model Components for Fine-Tuning:**
   - **Stable Diffusion Components:** Extracted `vae`, `text_encoder`, `tokenizer`, and `unet` from the `StableDiffusionPipeline` for separate fine-tuning.
   - **Set to Train Mode:** Ensured that `vae`, `text_encoder`, and `unet` are in training mode.

3. **Classification Model:**
   - Remains the same, but ensure that the input dimensions (`input_dim`) correctly reflect the embeddings' size from CLIP and VAE. Adjust `vae_dim` calculation based on your VAE architecture.

4. **Optimizers:**
   - **Classifier Optimizer (`optimizer_cls`):** Optimizes only the classifier parameters.
   - **Stable Diffusion Optimizer (`optimizer_sd`):** Optimizes the `unet` and `text_encoder` parameters. Fine-tuning VAE is optional and depends on your specific requirements.

5. **Training Loop Enhancements:**
   - **Batch Size:** Reduced to `4` due to high memory usage during fine-tuning. Adjust based on your GPU capacity.
   - **Loss Computation:**
     - **Diffusion Loss (`loss_unet`):** Uses Mean Squared Error (MSE) between predicted noise and actual noise. Adjust based on the loss function suitable for your diffusion model.
     - **Classification Loss (`loss_cls`):** Cross-entropy loss for artist classification.
     - **Combined Loss:** Sum of diffusion loss and classification loss. You can introduce weighting if necessary.
   - **Mixed Precision Training:** Utilizes PyTorch's `GradScaler` and `autocast` for efficient mixed-precision training to speed up and reduce memory usage.
   - **Gradients Zeroing:** Ensures that gradients are zeroed for both optimizers before the backward pass.

6. **Saving Models:**
   - **Classifier:** Saved using `safetensors`.
   - **Stable Diffusion Components:** Saved separately using `save_pretrained` for easier reloading and reuse.

7. **Uploading to Hugging Face Hub:**
   - **Separate Repositories:** Created separate repositories for the classifier and the fine-tuned Stable Diffusion model.
   - **Uploading:** Utilizes `upload_file` and `upload_folder` for uploading model files and directories.

8. **Inference Functions:**
   - **`generate_image_with_artist_reference`:** Generates an image based on a prompt and provides artist attribution.
   - **`verify_external_image_enhanced`:** Verifies an external image against the classifier to provide artist similarity scores.

9. **Example Usage:**
   - Demonstrates how to generate an image with attribution and verify an external image.

### **Important Considerations:**

- **Computational Resources:** Fine-tuning Stable Diffusion is resource-intensive. Ensure you have access to high-memory GPUs (e.g., NVIDIA A100) and consider reducing model sizes or batch sizes if necessary.
  
- **Dataset Quality:** The quality and relevance of your dataset's text prompts significantly impact the fine-tuning outcome. Ensure that prompts accurately describe the images and are diverse enough to generalize well.

- **Loss Balancing:** You might need to experiment with balancing the diffusion loss and classification loss, possibly introducing weights to prioritize one over the other based on your application needs.

- **Saving Checkpoints:** For longer training sessions, implement checkpoint saving to prevent loss of progress in case of interruptions.

- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and optimizer settings to achieve optimal performance.

### **Final Notes:**

Fine-tuning large models like Stable Diffusion requires careful handling to ensure both the generative capabilities and the classifier's accuracy are maintained and enhanced. The provided code serves as a foundational framework. Depending on your specific use case, further customization and optimization might be necessary.

Feel free to reach out if you encounter any challenges or have further questions regarding this implementation!
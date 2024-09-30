import os
import torch
from diffusers import StableDiffusionPipeline
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Pretrained Models
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
sd_pipeline = sd_pipeline.to(device)
vae = sd_pipeline.vae
vae.to(device)
vae.eval()

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
    return clip_features.cpu().numpy()

def get_vae_embedding(image, vae, vae_transform, device):
    image = vae_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        vae_features = latent_dist.mean
    return vae_features.cpu().numpy()

def get_combined_embedding(image, clip_processor, clip_model, vae, vae_transform, device):
    clip_emb = get_clip_embedding(image, clip_processor, clip_model, device)
    vae_emb = get_vae_embedding(image, vae, vae_transform, device)
    combined_emb = np.concatenate([clip_emb, vae_emb], axis=1)
    return combined_emb

# 4. Define Custom Dataset using Hugging Face datasets
class ArtistDataset(Dataset):
    def __init__(self, dataset_name, split, clip_processor, clip_model, vae, vae_transform, label_encoder):
        self.dataset = load_dataset(dataset_name, split=split)
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vae = vae
        self.vae_transform = vae_transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        artist = item['artist']

        # Get combined embedding
        combined_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vae, self.vae_transform, device)

        # Get label
        label = self.label_encoder.transform([artist])[0]

        return torch.tensor(combined_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 5. Initialize Label Encoder and Save
dataset_name = "your_username/your_dataset_name"  # Replace with your Hugging Face dataset name
dataset = load_dataset(dataset_name)

# Gather artist names
artist_names = dataset['train'].unique('artist')

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(artist_names)

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 6. Create Dataset and DataLoader
batch_size = 32
num_workers = 4

train_dataset = ArtistDataset(
    dataset_name=dataset_name,
    split='train',
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
clip_dim = 512
vae_dim = 512
input_dim = clip_dim + vae_dim

num_classes = len(artist_names)

model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# 8. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 9. Training Loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 10. Save the Trained Model using safetensors
save_file(model.state_dict(), "artist_classifier.safetensors")
print("Model saved to artist_classifier.safetensors")

# 11. Upload the model to Hugging Face Hub
api = HfApi()
repo_id = "your-username/your-model-name"  # Replace with your desired repository name

# Create the repository if it doesn't exist
api.create_repo(repo_id, exist_ok=True)

# Upload the model file
api.upload_file(
    path_or_fileobj="artist_classifier.safetensors",
    path_in_repo="artist_classifier.safetensors",
    repo_id=repo_id,
)

# Upload the label encoder
api.upload_file(
    path_or_fileobj="label_encoder.pkl",
    path_in_repo="label_encoder.pkl",
    repo_id=repo_id,
)

print(f"Model and label encoder uploaded to {repo_id}")

# 12. Load the Trained Model (for inference)
model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("artist_classifier.safetensors"))
model.eval()

# 13. Define Inference Functions
def generate_image_with_artist_reference(prompt, sd_pipeline, model, clip_processor, clip_model, vae, vae_transform, label_encoder, device, top_k=3):
    # Generate Image
    with torch.autocast(device.type):
        generated_image = sd_pipeline(prompt).images[0]

    # Get combined embedding
    combined_emb = get_combined_embedding(generated_image, clip_processor, clip_model, vae, vae_transform, device)
    combined_emb_tensor = torch.tensor(combined_emb, dtype=torch.float32).to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(combined_emb_tensor)
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
    combined_emb_tensor = torch.tensor(combined_emb, dtype=torch.float32).to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(combined_emb_tensor)
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
    # Example Prompt
    prompt = "A futuristic cityscape at sunset"
    generated_image, attribution = generate_image_with_artist_reference(
        prompt=prompt,
        sd_pipeline=sd_pipeline,
        model=model,
        clip_processor=clip_processor,
        clip_model=clip_model,
        vae=vae,
        vae_transform=vae_transform,
        label_encoder=label_encoder,
        device=device,
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

import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
from classifier import ArtistClassifier
from datasets import load_dataset
from safetensors.torch import save_file
from huggingface_hub import HfApi
from torch.cuda.amp import GradScaler, autocast

from utils import get_combined_embedding

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained Models
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# Move components to device
sd_pipeline.to(device)

# Extract components for fine-tuning
vae = sd_pipeline.vae
tokenizer = sd_pipeline.tokenizer

# Load CLIP model for classifier
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)

# Define Custom Dataset using Hugging Face datasets
class ArtistDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, clip_processor, clip_model, vae, label_encoder, max_length=77):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vae = vae
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
        combined_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vae, device)

        # Get label
        label = self.label_encoder.transform([artist])[0]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'combined_emb': combined_emb.squeeze(0),
            'label': label
        }

# Initialize Label Encoder and Save
dataset_name = "huggan/wikiart"  # Replace with your Hugging Face dataset name
dataset = load_dataset(dataset_name, split='train')

# Gather artist names
artist_names = dataset.unique('artist')

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(artist_names)

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Create Dataset and DataLoader
batch_size = 16 
num_workers = 4

train_dataset = ArtistDataset(
    dataset_name=dataset_name,
    split='train',
    tokenizer=tokenizer,
    clip_processor=clip_processor,
    clip_model=clip_model,
    vae=vae,
    label_encoder=label_encoder
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# Determine input dimensions
clip_dim = clip_model.config.projection_dim if hasattr(clip_model.config, 'projection_dim') else 512
vae_dim = vae.config.latent_channels * 64 * 64  # Adjust based on VAE architecture
input_dim = clip_dim + vae_dim

num_classes = len(artist_names)

model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# Define Loss and Optimizer
criterion_cls = nn.CrossEntropyLoss()

# Optimizer for classifier
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4)

# Mixed Precision and Gradient Scaler
scaler = GradScaler()

# Training Loop
num_epochs = 10 # Adjust based on your needs and resources

for epoch in range(num_epochs):
    model.train()
    total_loss_cls = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        combined_emb = batch['combined_emb'].to(device)
        label = batch['label'].to(device)

        # Zero the gradients
        optimizer_cls.zero_grad()

        with autocast():
            # Forward pass through classifier
            logits = model(combined_emb)
            loss_cls = criterion_cls(logits, label)

            # Weight the losses as needed
            total_loss = loss_cls

        # Backpropagation with mixed precision
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_cls)
        scaler.update()

        total_loss_cls += loss_cls.item()

    avg_loss_cls = total_loss_cls / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss Classifier: {avg_loss_cls:.4f}")

# Save the classification head
save_file(model.state_dict(), "artist_classifier.safetensors")
print("Classifier saved to artist_classifier.safetensors")

# Upload the Models to Hugging Face Hub
api = HfApi()
repo_id_classifier = "your-username/artist-classifier"  # Replace with your repository name for classifier

# Create repositories if they don't exist
api.create_repo(repo_id_classifier, exist_ok=True)

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

print(f"Models uploaded to {repo_id_classifier}")
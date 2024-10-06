import torch
from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor, AutoModel, CLIPImageProcessor
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

from utils import assign_artist_to_code, find_nearest_nth_root_and_factors, generate_alphabet_sequence, generate_unique_code, get_combined_embedding

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained Models
flux_model_id = "black-forest-labs/FLUX.1-schnell"
flux_pipeline = FluxPipeline.from_pretrained(
    flux_model_id,
    torch_dtype=torch.bfloat16
)

# Move components to device
flux_pipeline.to(device)

# Extract VAE components
vae = flux_pipeline.vae

# Load CLIP model for classifier
clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float32)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)

# Load InternVIT model for classifier
vit_model_id = "OpenGVLab/InternViT-6B-448px-V1-5"
vit_model = AutoModel.from_pretrained(vit_model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
vit_processor = CLIPImageProcessor.from_pretrained(vit_model_id)
vit_model = vit_model.to(device)

# Define Custom Dataset using Hugging Face datasets
class ArtistDataset(Dataset):
    def __init__(self, dataset_name, split, clip_processor, clip_model, vit_processor, vit_model, vae, label_encoder_1, label_encoder_2, label_encoder_3, assigned_artists):
        self.dataset = load_dataset(dataset_name, split=split)
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vit_processor = vit_processor,
        self.vit_model = vit_model
        self.vae = vae
        self.label_encoder_1 = label_encoder_1
        self.label_encoder_2 = label_encoder_2
        self.label_encoder_3 = label_encoder_3
        self.assigned_artists = assigned_artists

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        artist = item['artist']
        unique_code = self.assigned_artists[artist]
        codes = unique_code.split("-")

        # Get combined embedding for classifier
        vae_emb, vit_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vit_processor, self.vit_model, self.vae, device)

        # Get label
        label_1 = self.label_encoder_1.transform([codes[0]])[0]
        label_2 = self.label_encoder_2.transform([codes[1]])[0]
        label_3 = self.label_encoder_3.transform([codes[2]])[0]

        return {
            'vae_emb': vae_emb.squeeze(0),
            'vit_emb': vit_emb.squeeze(0),
            'label_1': label_1,
            'label_2': label_2,
            'label_3': label_3,
        }

# Initialize Label Encoder and Save
dataset_name = "huggan/wikiart"  # Replace with your Hugging Face dataset name
dataset = load_dataset(dataset_name, split='train')

# Gather artist names
artist_names = dataset.unique('artist')

num_classes = len(artist_names)

classes_factors = find_nearest_nth_root_and_factors(num_classes)
classes_alphabet_sequence = generate_alphabet_sequence(classes_factors)
classes_unique_code = generate_unique_code(classes_alphabet_sequence)
assigned_artists = assign_artist_to_code(artist_names, classes_unique_code)

# Initialize and fit LabelEncoder
label_encoder_1 = LabelEncoder()
label_encoder_1.fit(classes_alphabet_sequence[0])

label_encoder_2 = LabelEncoder()
label_encoder_2.fit(classes_alphabet_sequence[1])

label_encoder_3 = LabelEncoder()
label_encoder_3.fit(classes_alphabet_sequence[2])

# Save LabelEncoder
with open("label_encoder_1.pkl", "wb") as f:
    pickle.dump(label_encoder_1, f)

with open("label_encoder_2.pkl", "wb") as f:
    pickle.dump(label_encoder_2, f)

with open("label_encoder_3.pkl", "wb") as f:
    pickle.dump(label_encoder_3, f)

# Create Dataset and DataLoader
batch_size = 16 
num_workers = 4

train_dataset = ArtistDataset(
    dataset_name=dataset_name,
    split='train',
    clip_processor=clip_processor,
    clip_model=clip_model,
    vit_processor=vit_processor,
    vit_model=vit_model,
    vae=vae,
    label_encoder_1=label_encoder_1,
    label_encoder_2=label_encoder_2,
    label_encoder_3=label_encoder_3,
    assigned_artists=assigned_artists
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# Determine input dimensions
clip_dim = clip_model.config.projection_dim if hasattr(clip_model.config, 'projection_dim') else 512
vit_dim = vit_model.config.hidden_size if hasattr(clip_model.config, 'hidden_size') else 512

vae_dim = vae.config.latent_channels * 64 * 64 # Adjust based on VAE architecture and VAE transformation

model = ArtistClassifier(vae_dim=vae_dim, vit_dim=(clip_dim + vit_dim), classes_factors=classes_factors).to(device)

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
        vae_emb = batch['vae_emb'].to(device)
        vit_emb = batch['vit_emb'].to(device)
        label_1 = batch['label_1'].to(device)
        label_2 = batch['label_2'].to(device)
        label_3 = batch['label_3'].to(device)

        # Zero the gradients
        optimizer_cls.zero_grad()

        with autocast():
            # Forward pass through classifier
            logits_1, logits_2, logits_3 = model(vae_emb, vit_emb)

            loss_cls_1 = criterion_cls(logits_1, label_1)
            loss_cls_2 = criterion_cls(logits_2, label_2)
            loss_cls_3 = criterion_cls(logits_3, label_3)

            # Weight the losses as needed
            total_loss = loss_cls_1 + loss_cls_2 + loss_cls_3

        # Backpropagation with mixed precision
        scaler.scale(total_loss).backward()
        scaler.step(optimizer_cls)
        scaler.update()

        total_loss_cls += loss_cls_1.item()
        total_loss_cls += loss_cls_2.item()
        total_loss_cls += loss_cls_3.item()

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
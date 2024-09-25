# Lawful Diffusion

This implementation provides a comprehensive framework for integrating **Stable Diffusion** with a **retrieval-based attribution system** using **PyTorch**, **Hugging Face's Diffusers**, and **CLIP**. By encoding and indexing the training dataset's images, the system can both attribute generated images and verify external images against the training data.

This approach promotes transparency and accountability in generative models, addressing concerns related to copyright and artist attribution. It serves as a foundation that can be further refined and expanded based on specific requirements and datasets.

---

## Goals

Building a sophisticated generative model architecture that integrates **Stable Diffusion** with a **retrieval-based attribution system** involves several components. This system will not only generate images based on text prompts but also provide attributions to the artists or data sources that most closely align with the generated content. Additionally, it will offer verification capabilities for external images against the training dataset.

---

A **training pipeline** that allows a generative model like **Stable Diffusion** to **output the nearest artist reference text based on both CLIP embeddings and autoencoder (VAE) embeddings** involves the following key steps:

1. **Data Preparation**: Organize dataset with images and associated artist labels.
2. **Embedding Extraction**:
   - **CLIP Embeddings**: Encode images using the CLIP model.
   - **Autoencoder (VAE) Embeddings**: Extract latent representations from the VAE part of the Stable Diffusion model.
3. **Combining Embeddings**: Merge CLIP and VAE embeddings to create a comprehensive representation.
4. **Label Encoding**: Encode artist labels for training.
5. **Model Training**: Train a classifier (e.g., a neural network) to predict artist labels based on combined embeddings.
6. **Integration with Generation Pipeline**: Enhance the image generation process to output artist references alongside generated images.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#1-data-preparation)
3. [Load Pretrained Models](#2-load-pretrained-models)
4. [Extract and Combine Embeddings](#3-extract-and-combine-embeddings)
5. [Encode Labels](#4-encode-labels)
6. [Create Dataset and DataLoader](#5-create-dataset-and-dataloader)
7. [Define the Classification Model](#6-define-the-classification-model)
8. [Train the Model](#7-train-the-model)
9. [Save and Load the Trained Model](#8-save-and-load-the-trained-model)
10. [Integrate with Image Generation Pipeline](#9-integrate-with-image-generation-pipeline)
11. [External Image Verification Enhancement](#10-external-image-verification-enhancement)
12. [Complete Code Example](#complete-code-example)

![Architecture Diagram](flowchart.png)

---

## Prerequisites

Ensure you have the following libraries installed:

```bash
pip install torch torchvision transformers diffusers faiss-cpu pillow scikit-learn
# For GPU support with FAISS:
pip install faiss-gpu
```

*Note: Adjust the FAISS installation based on your system's compatibility.*

---

## 1. Data Preparation

Organize your dataset in a structured manner. Each image should be associated with an artist label. Here's a sample structure:

```
dataset/
    artist1/
        image1.jpg
        image2.jpg
        ...
    artist2/
        image1.jpg
        ...
    ...
```

Ensure that each artist has a sufficient number of images to train the classifier effectively.

---

## 2. Load Pretrained Models

We'll utilize the **Stable Diffusion** pipeline for image generation and the **CLIP** model for encoding.

```python
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Stable Diffusion Pipeline
sd_model_id = "runwayml/stable-diffusion-v1-5"  # You can choose different variants
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
sd_pipeline = sd_pipeline.to(device)

# Extract VAE from Stable Diffusion
vae = sd_pipeline.vae
vae.to(device)
vae.eval()

# Load CLIP Model and Processor
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)
clip_model.eval()
```

---

## 3. Extract and Combine Embeddings

We'll extract embeddings from both CLIP and the VAE to create a combined representation for each image.

### 3.1. Extract CLIP Embeddings

```python
def get_clip_embedding(image, clip_processor, clip_model, device):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**inputs)
    clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)
    return clip_features.cpu().numpy()
```

### 3.2. Extract VAE Embeddings

Stable Diffusion's VAE can be used to encode images into latent space.

```python
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Define the transformation pipeline as per VAE requirements
vae_transform = Compose([
    Resize(sd_pipeline.unet.config.sample_size * 8, interpolation=Image.BICUBIC),
    CenterCrop(sd_pipeline.unet.config.sample_size * 8),
    ToTensor(),
    Normalize([0.5], [0.5])
])

def get_vae_embedding(image, vae, vae_transform, device):
    # Transform the image
    image = vae_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        vae_features = latent_dist.mean
    return vae_features.cpu().numpy()
```

### 3.3. Combine CLIP and VAE Embeddings

We'll concatenate the CLIP and VAE embeddings to form a unified representation.

```python
import numpy as np

def get_combined_embedding(image, clip_processor, clip_model, vae, vae_transform, device):
    clip_emb = get_clip_embedding(image, clip_processor, clip_model, device)
    vae_emb = get_vae_embedding(image, vae, vae_transform, device)
    combined_emb = np.concatenate([clip_emb, vae_emb], axis=1)
    return combined_emb
```

*Note: Ensure both embeddings have compatible dimensions for concatenation.*

---

## 4. Encode Labels

Convert artist labels into numerical format using label encoding.

```python
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image

# Assuming you have a list of artist names
dataset_path = "path_to_your_dataset"  # Replace with your dataset path

# Gather all artist names
artist_names = [artist for artist in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, artist))]

# Initialize LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(artist_names)

# Example:
# labels = label_encoder.transform(['artist1', 'artist2', ...])
```

*Save the label encoder for future use:*

```python
import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
```

*To load later:*

```python
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
```

---

## 5. Create Dataset and DataLoader

Define a custom PyTorch dataset to handle image loading, embedding extraction, and label encoding.

```python
from torch.utils.data import Dataset, DataLoader

class ArtistDataset(Dataset):
    def __init__(self, dataset_path, clip_processor, clip_model, vae, vae_transform, label_encoder, transform=None):
        self.dataset_path = dataset_path
        self.artists = [artist for artist in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, artist))]
        self.data = []
        for artist in self.artists:
            artist_path = os.path.join(dataset_path, artist)
            for img_file in os.listdir(artist_path):
                img_path = os.path.join(artist_path, img_file)
                self.data.append((img_path, artist))
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vae = vae
        self.vae_transform = vae_transform
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, artist = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Handle corrupted images as needed
            return None

        # Get combined embedding
        combined_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vae, self.vae_transform, device)

        # Get label
        label = self.label_encoder.transform([artist])[0]

        return torch.tensor(combined_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
```

**Create DataLoader:**

```python
batch_size = 32
num_workers = 4  # Adjust based on your system

dataset = ArtistDataset(
    dataset_path=dataset_path,
    clip_processor=clip_processor,
    clip_model=clip_model,
    vae=vae,
    vae_transform=vae_transform,
    label_encoder=label_encoder
)

# Filter out None samples (if any)
filtered_data = [sample for sample in dataset if sample is not None]
combined_features, labels = zip(*filtered_data)

combined_features = torch.stack(combined_features)
labels = torch.stack(labels)

# Create a new dataset from the filtered data
filtered_dataset = torch.utils.data.TensorDataset(combined_features, labels)

# DataLoader
dataloader = DataLoader(
    filtered_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
```

*Note: For large datasets, consider streaming data or using more efficient data handling techniques.*

---

## 6. Define the Classification Model

We'll define a simple neural network classifier that takes combined embeddings as input and outputs artist labels.

```python
import torch.nn as nn
import torch.nn.functional as F

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
```

**Determine Input Dimensions:**

- **CLIP Embedding Dimension**: Typically 512 for `clip-vit-base-patch32`.
- **VAE Latent Dimension**: This varies based on the VAE's configuration. For Stable Diffusion's `v1-5`, it's usually 4 times hidden dimension.

Assuming:

```python
clip_dim = 512
vae_dim = 512  # Adjust based on actual VAE configuration
input_dim = clip_dim + vae_dim  # 1024
```

*Verify dimensions based on your specific model's configuration.*

---

## 7. Train the Model

Define the training loop, loss function, optimizer, and train the classifier.

```python
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Initialize model
num_classes = len(artist_names)
model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training parameters
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
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

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Optional: Validation step
    # You can split your dataset into train and validation sets
    # and compute validation accuracy here.
```

*Enhancements:*

- **Validation Split**: Split dataset into training and validation sets to monitor performance.
- **Learning Rate Scheduling**: Adjust learning rate based on validation performance.
- **Early Stopping**: Stop training when validation performance stops improving.

---

## 8. Save and Load the Trained Model

**Save the Model:**

```python
torch.save(model.state_dict(), "artist_classifier.pth")
```

**Load the Model:**

```python
model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("artist_classifier.pth"))
model.eval()
```

---

## 9. Integrate with Image Generation Pipeline

After training the classifier, integrate it with the image generation pipeline to output artist references based on generated images' embeddings.

```python
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
```

**Usage Example:**

```python
prompt = "A surreal landscape with floating islands and waterfalls"
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
print("Attribution:")
for attrib in attribution:
    print(f"Artist: {attrib['artist']}, Probability: {attrib['probability']:.2f}")
```

---

## 10. External Image Verification Enhancement

Enhance the verification module to utilize both CLIP and VAE embeddings for improved accuracy in matching external images to training data.

### 10.1. Extend Verification to Use Combined Embeddings

```python
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
```

**Usage Example:**

```python
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
    print("Verification Report:")
    for idx, report in enumerate(verification_report, 1):
        print(f"{idx}. Artist: {report['artist']}, Probability: {report['probability']:.2f}")
else:
    print("No similar images found or error in processing.")
```

*Note: The verification now leverages the classifier's predictions based on combined embeddings, providing a probabilistic attribution to potential artists.*

---

## 11. Complete Code Example

Below is a consolidated script that brings together all the steps mentioned above. Ensure you replace placeholders like `path_to_your_dataset` with actual paths.

```python
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
    Resize(512, interpolation=Image.BICUBIC),  # Adjust based on your model
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

# 4. Define Custom Dataset
class ArtistDataset(Dataset):
    def __init__(self, dataset_path, clip_processor, clip_model, vae, vae_transform, label_encoder, transform=None):
        self.dataset_path = dataset_path
        self.artists = [artist for artist in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, artist))]
        self.data = []
        for artist in self.artists:
            artist_path = os.path.join(dataset_path, artist)
            for img_file in os.listdir(artist_path):
                img_path = os.path.join(artist_path, img_file)
                self.data.append((img_path, artist))
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.vae = vae
        self.vae_transform = vae_transform
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, artist = self.data[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Handle corrupted images as needed
            return None

        # Get combined embedding
        combined_emb = get_combined_embedding(image, self.clip_processor, self.clip_model, self.vae, self.vae_transform, device)

        # Get label
        label = self.label_encoder.transform([artist])[0]

        return torch.tensor(combined_emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 5. Initialize Label Encoder and Save
dataset_path = "path_to_your_dataset"  # Replace with your dataset path

# Gather artist names
artist_names = [artist for artist in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, artist))]

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(artist_names)

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 6. Create Dataset and DataLoader
batch_size = 32
num_workers = 4  # Adjust based on your system

dataset = ArtistDataset(
    dataset_path=dataset_path,
    clip_processor=clip_processor,
    clip_model=clip_model,
    vae=vae,
    vae_transform=vae_transform,
    label_encoder=label_encoder
)

# Filter out None samples (if any)
filtered_data = [sample for sample in dataset if sample is not None]
combined_features, labels = zip(*filtered_data)

combined_features = torch.stack(combined_features)
labels = torch.stack(labels)

# Create a new dataset from the filtered data
filtered_dataset = torch.utils.data.TensorDataset(combined_features, labels)

# DataLoader
dataloader = DataLoader(
    filtered_dataset,
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
vae_dim = 512  # Adjust based on actual VAE configuration
input_dim = clip_dim + vae_dim  # 1024

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
    for batch in dataloader:
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

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Optional: Validation can be added here

# 10. Save the Trained Model
torch.save(model.state_dict(), "artist_classifier.pth")
print("Model saved to artist_classifier.pth")

# 11. Load the Trained Model (for inference)
model = ArtistClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("artist_classifier.pth"))
model.eval()

# 12. Define Inference Functions
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

# 13. Example Usage
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
```

**Explanation of the Complete Script:**

1. **Model Loading**: Loads Stable Diffusion and CLIP models.
2. **Embedding Extraction**: Defines functions to extract both CLIP and VAE embeddings and combine them.
3. **Dataset Definition**: Creates a custom `ArtistDataset` to handle data loading and embedding extraction.
4. **Label Encoding**: Uses sklearn's `LabelEncoder` to convert artist names to numerical labels.
5. **DataLoader Creation**: Prepares a `DataLoader` for efficient batching during training.
6. **Model Definition**: Defines a simple neural network classifier with two hidden layers.
7. **Training Loop**: Trains the classifier using cross-entropy loss and the Adam optimizer.
8. **Model Saving and Loading**: Provides mechanisms to save the trained model and reload it for inference.
9. **Inference Functions**: Defines functions to generate images with artist attributions and to verify external images.
10. **Example Usage**: Demonstrates how to use the trained model for generating images and verifying external images.

---

**Key Considerations:**

1. **Data Quality and Quantity**: Ensure a diverse and sufficiently large dataset for robust training.
2. **Embedding Dimensions**: Verify the dimensions of both CLIP and VAE embeddings to ensure proper concatenation.
3. **Model Complexity**: Adjust the classifier's complexity based on dataset size and diversity to prevent overfitting or underfitting.
4. **Performance Optimization**: Utilize GPU acceleration where possible, especially for embedding extraction and model training.
5. **Ethical and Legal Compliance**: Ensure that all data used respects copyright and licensing agreements.

**Potential Enhancements:**

- **Multi-label Classification**: If images can belong to multiple artists or styles, modify the classifier to handle multi-label predictions.
- **Clustering**: Use unsupervised learning to discover artist clusters without predefined labels.
- **Fine-Tuning Models**: Fine-tune the CLIP or VAE models specifically for your dataset to improve embedding representations.

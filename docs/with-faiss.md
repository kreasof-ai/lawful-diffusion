# Lawful Diffusion

This implementation provides a comprehensive framework for integrating **Stable Diffusion** with a **retrieval-based attribution system** using **PyTorch**, **Hugging Face's Diffusers**, **CLIP**, and **FAISS**. By encoding and indexing the training dataset's images, the system can both attribute generated images and verify external images against the training data.

This approach promotes transparency and accountability in generative models, addressing concerns related to copyright and artist attribution. It serves as a foundation that can be further refined and expanded based on specific requirements and datasets.

---

## Goals

Building a sophisticated generative model architecture that integrates **Stable Diffusion** with a **retrieval-based attribution system** involves several components. This system will not only generate images based on text prompts but also provide attributions to the artists or data sources that most closely align with the generated content. Additionally, it will offer verification capabilities for external images against the training dataset.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Implementation](#step-by-step-implementation)
    - [1. Setup and Installations](#1-setup-and-installations)
    - [2. Load Pretrained Models](#2-load-pretrained-models)
    - [3. Prepare the Training Data and Create Index](#3-prepare-the-training-data-and-create-index)
    - [4. Implement the Image Generation with Attribution](#4-implement-the-image-generation-with-attribution)
    - [5. External Image Verification](#5-external-image-verification)
4. [Complete Code Example](#complete-code-example)
5. [Conclusion](#conclusion)

---

## Architecture Overview

The proposed architecture consists of the following key components:

1. **Stable Diffusion Model**: Generates images based on textual prompts.
2. **CLIP Model**: Encodes both images and text into a shared embedding space.
3. **FAISS Index**: Facilitates efficient similarity searches within the embedding space.
4. **Attribution System**: Retrieves the most similar artists or data sources based on the generated image's embedding.
5. **Verification Module**: Checks external images against the training dataset to verify their origin or similarity.

---

## Prerequisites

Before diving into the implementation, ensure you have the following installed:

- **Python 3.8+**
- **PyTorch**
- **Hugging Face Transformers and Diffusers**
- **FAISS**
- **Other Dependencies**: Such as `numpy`, `PIL`, etc.

You can install the necessary libraries using `pip`:

```bash
pip install torch torchvision transformers diffusers faiss-cpu faiss-gpu
```

*Note: Depending on your system and requirements, you might prefer `faiss-gpu` for better performance if you have a compatible GPU.*

---

## Step-by-Step Implementation

### 1. Setup and Installations

Ensure all necessary libraries are installed. Here's a quick installation guide:

```bash
pip install torch torchvision transformers diffusers faiss-cpu
```

For GPU acceleration with FAISS:

```bash
pip install faiss-gpu
```

### 2. Load Pretrained Models

We'll use the **Stable Diffusion** model for image generation and **CLIP** for encoding images and texts into embeddings.

```python
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
import faiss
import numpy as np
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Load the Stable Diffusion model:

```python
# Load Stable Diffusion Pipeline
sd_model_id = "runwayml/stable-diffusion-v1-5"  # You can choose different variants
sd = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
sd = sd.to(device)
```

Load the CLIP model and processor:

```python
# Load CLIP Model
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)
clip_model.eval()
```

### 3. Prepare the Training Data and Create Index

Assume you have a dataset of artist images or reference images, each associated with a label (e.g., artist name).

**Dataset Structure:**

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

**Steps:**

1. **Load and Encode Images:** Use CLIP to encode each image into an embedding.
2. **Build FAISS Index:** Create an index of these embeddings for efficient similarity search.
3. **Maintain Metadata:** Keep a mapping from index positions to metadata (e.g., artist names).

```python
# Paths
dataset_path = "path_to_your_dataset"  # Replace with your dataset path

# Initialize lists
image_embeddings = []
metadata = []

# Function to encode images
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding /= embedding.norm(pdim=1, keepdim=True)
    return embedding.cpu().numpy()

# Iterate through dataset and encode images
for artist_folder in os.listdir(dataset_path):
    artist_path = os.path.join(dataset_path, artist_folder)
    if not os.path.isdir(artist_path):
        continue
    for img_file in os.listdir(artist_path):
        img_path = os.path.join(artist_path, img_file)
        try:
            emb = encode_image(img_path)
            image_embeddings.append(emb)
            metadata.append({
                "artist": artist_folder,
                "image_path": img_path
            })
        except Exception as e:
            print(f"Error encoding {img_path}: {e}")

# Convert list to numpy array
image_embeddings = np.vstack(image_embeddings).astype('float32')

# Create FAISS index
dimension = image_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance
index.add(image_embeddings)

# Save index and metadata for future use
faiss.write_index(index, "image_index.faiss")
import pickle
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
```

*Note: Encoding a large dataset can be time-consuming. Consider saving the index and metadata for reuse.*

Load the index and metadata when needed:

```python
# Load index and metadata
index = faiss.read_index("image_index.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
```

### 4. Implement the Image Generation with Attribution

When generating an image based on a text prompt, we'll:

1. **Generate the Image:** Use Stable Diffusion.
2. **Encode the Image:** Use CLIP to get its embedding.
3. **Retrieve Similar Images:** Query the FAISS index to find similar images.
4. **Extract Attribution:** Based on the retrieved similar images' metadata.

```python
def generate_image_with_attribution(prompt, num_retrieval=5):
    # Generate Image
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = sd(prompt).images[0]
    
    # Encode Generated Image
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_embedding = clip_model.get_image_features(**inputs)
    generated_embedding /= generated_embedding.norm(pdim=1, keepdim=True)
    generated_embedding = generated_embedding.cpu().numpy().astype('float32')
    
    # Retrieve Similar Images
    distances, indices = index.search(generated_embedding, num_retrieval)
    
    # Gather Attribution Info
    attribution = []
    for idx in indices[0]:
        attrib = metadata[idx]
        attribution.append({
            "artist": attrib["artist"],
            "image_path": attrib["image_path"]
        })
    
    return image, attribution
```

**Usage Example:**

```python
prompt = "A surreal landscape with floating islands and waterfalls"
generated_image, attribution = generate_image_with_attribution(prompt)

# Display the image
generated_image.show()

# Print attribution
print("Attribution:")
for attrib in attribution:
    print(f"Artist: {attrib['artist']}, Image Path: {attrib['image_path']}")
```

### 5. External Image Verification

To verify an external image against the training dataset:

1. **Encode the External Image:** Using CLIP.
2. **Search Similar Images:** Use FAISS to find similar images in the index.
3. **Determine Similarity:** Based on distance or similarity scores.
4. **Provide Verification Report:** Including attribution if similar.

```python
def verify_external_image(image_path, threshold=0.3, num_retrieval=5):
    # Encode External Image
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            external_embedding = clip_model.get_image_features(**inputs)
        external_embedding /= external_embedding.norm(pdim=1, keepdim=True)
        external_embedding = external_embedding.cpu().numpy().astype('float32')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None
    
    # Retrieve Similar Images
    distances, indices = index.search(external_embedding, num_retrieval)
    
    # Determine if similar based on threshold
    similar = []
    for distance, idx in zip(distances[0], indices[0]):
        # Convert L2 distance to cosine similarity
        similarity = 1 - distance / 2  # Since embeddings are normalized
        if similarity >= (1 - threshold):
            attrib = metadata[idx]
            similar.append({
                "artist": attrib["artist"],
                "image_path": attrib["image_path"],
                "similarity": similarity
            })
    
    return similar
```

**Usage Example:**

```python
external_image_path = "path_to_external_image.jpg"
similar_images = verify_external_image(external_image_path)

if similar_images:
    print("The external image is similar to the following dataset images:")
    for sim in similar_images:
        print(f"Artist: {sim['artist']}, Image Path: {sim['image_path']}, Similarity: {sim['similarity']:.2f}")
else:
    print("No similar images found in the dataset.")
```

---

## Complete Code Example

Below is the complete code encapsulating all the steps mentioned above. This example assumes you have a dataset structured as described and that you're using a machine with sufficient computational resources.

```python
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
import faiss
import numpy as np
from PIL import Image
import os
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Stable Diffusion Pipeline
sd_model_id = "runwayml/stable-diffusion-v1-5"
sd = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
sd = sd.to(device)

# Load CLIP Model
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)
clip_model.eval()

# Paths
dataset_path = "path_to_your_dataset"  # Replace with your dataset path

# Initialize lists
image_embeddings = []
metadata = []

# Function to encode images
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding /= embedding.norm(pdim=1, keepdim=True)
    return embedding.cpu().numpy()

# Encode dataset images
print("Encoding dataset images...")
for artist_folder in os.listdir(dataset_path):
    artist_path = os.path.join(dataset_path, artist_folder)
    if not os.path.isdir(artist_path):
        continue
    for img_file in os.listdir(artist_path):
        img_path = os.path.join(artist_path, img_file)
        try:
            emb = encode_image(img_path)
            image_embeddings.append(emb)
            metadata.append({
                "artist": artist_folder,
                "image_path": img_path
            })
        except Exception as e:
            print(f"Error encoding {img_path}: {e}")

# Convert list to numpy array
image_embeddings = np.vstack(image_embeddings).astype('float32')

# Create FAISS index
dimension = image_embeddings.shape[1]
print("Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings)

# Save index and metadata
print("Saving index and metadata...")
faiss.write_index(index, "image_index.faiss")
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Function to load index and metadata
def load_index_metadata(index_path="image_index.faiss", metadata_path="metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load index and metadata
index, metadata = load_index_metadata()

# Function to generate image with attribution
def generate_image_with_attribution(prompt, num_retrieval=5):
    # Generate Image
    print(f"Generating image for prompt: '{prompt}'")
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = sd(prompt).images[0]
    
    # Encode Generated Image
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_embedding = clip_model.get_image_features(**inputs)
    generated_embedding /= generated_embedding.norm(pdim=1, keepdim=True)
    generated_embedding = generated_embedding.cpu().numpy().astype('float32')
    
    # Retrieve Similar Images
    distances, indices = index.search(generated_embedding, num_retrieval)
    
    # Gather Attribution Info
    attribution = []
    for idx in indices[0]:
        attrib = metadata[idx]
        attribution.append({
            "artist": attrib["artist"],
            "image_path": attrib["image_path"]
        })
    
    return image, attribution

# Function to verify external image
def verify_external_image(image_path, threshold=0.3, num_retrieval=5):
    # Encode External Image
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            external_embedding = clip_model.get_image_features(**inputs)
        external_embedding /= external_embedding.norm(pdim=1, keepdim=True)
        external_embedding = external_embedding.cpu().numpy().astype('float32')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None
    
    # Retrieve Similar Images
    distances, indices = index.search(external_embedding, num_retrieval)
    
    # Determine if similar based on threshold
    similar = []
    for distance, idx in zip(distances[0], indices[0]):
        # Convert L2 distance to cosine similarity
        similarity = 1 - distance / 2  # Since embeddings are normalized
        if similarity >= (1 - threshold):
            attrib = metadata[idx]
            similar.append({
                "artist": attrib["artist"],
                "image_path": attrib["image_path"],
                "similarity": similarity
            })
    
    return similar

# Example Usage
if __name__ == "__main__":
    # Generate Image with Attribution
    prompt = "A futuristic cityscape at sunset"
    generated_image, attribution = generate_image_with_attribution(prompt)
    
    # Display the generated image
    generated_image.show()
    
    # Print attribution
    print("\nAttribution:")
    for idx, attrib in enumerate(attribution, 1):
        print(f"{idx}. Artist: {attrib['artist']}, Image Path: {attrib['image_path']}")
    
    # Verify External Image
    external_image_path = "path_to_external_image.jpg"  # Replace with your image path
    similar_images = verify_external_image(external_image_path)
    
    if similar_images:
        print("\nThe external image is similar to the following dataset images:")
        for sim in similar_images:
            print(f"Artist: {sim['artist']}, Image Path: {sim['image_path']}, Similarity: {sim['similarity']:.2f}")
    else:
        print("\nNo similar images found in the dataset.")
```

**Key Points in the Complete Code:**

- **Dataset Encoding:** The script encodes all images in the specified dataset using CLIP and builds a FAISS index for similarity search.
  
- **Image Generation with Attribution:** When a prompt is given, the model generates an image, encodes it, and retrieves the most similar images from the dataset to provide attribution.
  
- **External Image Verification:** Given an external image, the system encodes it and checks against the FAISS index to determine if it's similar to any images in the training dataset.

**Important Considerations:**

1. **Storage and Performance:**
   - **FAISS Index Size:** For large datasets, consider using more efficient FAISS indexes like `IndexIVFFlat` or `IndexHNSWFlat` to balance speed and memory usage.
   - **Embedding Storage:** Storing high-dimensional embeddings can consume significant memory. Optimize by using appropriate FAISS index types.

2. **Dataset Licensing and Permissions:**
   - Ensure that all images used in the training dataset have the necessary permissions and licenses for use in this manner.

3. **Scalability:**
   - For very large datasets, consider distributed FAISS indexing or other scalable vector search solutions.

4. **Privacy and Security:**
   - Implement measures to protect the dataset's privacy and prevent unauthorized access to the embeddings and metadata.

5. **Threshold Tuning:**
   - The similarity threshold in the verification function (`threshold=0.3`) may need adjustment based on the specific dataset and use case.

6. **Error Handling:**
   - Robust error handling ensures that corrupted images or unexpected issues don't halt the entire process.

---
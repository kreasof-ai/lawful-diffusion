import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Training purposes

def get_clip_embedding(image, clip_processor, clip_model, device):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_features = clip_model.get_image_features(**inputs)
    clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)

    return clip_features.cpu()

def get_vit_embedding(image, vit_processor, vit_model, device):
    pixel_values = vit_processor(images=image, return_tensors='pt').to(device).pixel_values

    with torch.no_grad():
        vit_features = vit_model(pixel_values)

    return vit_features.cpu()

def get_vae_embedding(image, vae, device):
    # transform image size to 1024 * 1024
    vae_transform = Compose([
        Resize(1024, interpolation=Image.BICUBIC),
        CenterCrop(1024),
        ToTensor(),
        Normalize([0.5], [0.5])
    ]) 

    image = vae_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        vae_features = latent_dist.mean
    return vae_features.cpu()

def get_combined_embedding(image, clip_processor, clip_model, vit_processor, vit_model, vae, device):
    clip_emb = get_clip_embedding(image, clip_processor, clip_model, device)
    vit_emb = get_vit_embedding(image, vit_processor, vit_model, device)
    vae_emb = get_vae_embedding(image, vae, device)
    combined_emb = torch.cat([clip_emb, vit_emb], dim=1)
    return vae_emb, combined_emb

# Inference purposes

def generate_image_with_artist_reference(prompt, flux_pipeline, model, clip_processor, clip_model, vit_processor, vit_model, vae, label_encoder, device, top_k=3):
    # Generate Image
    with torch.no_grad():
        with torch.autocast(device.type):
            generated_image = flux_pipeline(prompt).images[0]

    # Get combined embedding
    vae_emb, vit_emb = get_combined_embedding(generated_image, clip_processor, clip_model, vit_processor, vit_model, vae, device)
    
    vit_emb_tensor = vit_emb.to(device)
    vae_emb_tensor = vae_emb.to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(vae_emb_tensor.unsqueeze(0), vit_emb_tensor.unsqueeze(0))
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

def verify_external_image_enhanced(image_path, model, clip_processor, clip_model, vit_processor, vit_model, vae, device, label_encoder, top_k=5):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Get combined embedding
    vae_emb, vit_emb = get_combined_embedding(image, clip_processor, clip_model, vit_processor, vit_model, vae, device)
    
    vit_emb_tensor = vit_emb.to(device)
    vae_emb_tensor = vae_emb.to(device)

    # Predict artist labels
    with torch.no_grad():
        logits = model(vae_emb_tensor.unsqueeze(0), vit_emb_tensor.unsqueeze(0))
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
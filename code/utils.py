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

def find_nearest_nth_root_and_factors(X, n=3):
    # Step 1: Find the nearest n-th root of X
    nth_root_X = X ** (1 / n)
    R = round(nth_root_X)

    # Step 2: Initialize all factors to R
    factors = [R] * n

    # Step 3: Adjust the factors to make their product at least X
    def product(factors):
        result = 1
        for factor in factors:
            result *= factor
        return result

    while product(factors) < X:
        for i in range(n):
            if product(factors) >= X:
                break
            factors[i] += 1

    return factors

def int_to_alphabet(num):
    if num <= 0:
        return ""
    result = []
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        result.append(chr(ord('A') + remainder))
    return ''.join(reversed(result))

def alphabet_loop(num):
    result = []
    for i in range(num):
        result.append(int_to_alphabet(i + 1))
    return result
    
def generate_alphabet_sequence(factors):
    return [alphabet_loop(factor) for factor in factors]

def generate_unique_code(alphabet_sequence):
    sequence_1 = alphabet_sequence[0]
    sequence_2 = alphabet_sequence[1]
    sequence_3 = alphabet_sequence[2] # Only works for N number of factors
    
    result = []
    
    for i in range(len(sequence_1)):
        tmp_1 = sequence_1[i]
        
        for j in range(len(sequence_2)):
            tmp_2 = sequence_2[j]
            
            for k in range(len(sequence_3)):
                tmp_3 = sequence_3[k]
                
                result.append(tmp_1 + "-" + tmp_2 + "-" + tmp_3)
                
    return result

def assign_code_to_artists(artist_names, unique_code, alphabet_sequence):
    table_1 = {}
    table_2 = {}
    table_3 = {}

    for i in range(len(alphabet_sequence[0])):
        table_1[alphabet_sequence[0][i]] = []
    
    for i in range(len(alphabet_sequence[1])):
        table_2[alphabet_sequence[1][i]] = []
    
    for i in range(len(alphabet_sequence[2])):
        table_3[alphabet_sequence[2][i]] = []

    for i in range(len(artist_names)):
        codes = unique_code[i].split("-")

        table_1[codes[0]].append(artist_names[i])
        table_2[codes[1]].append(artist_names[i])
        table_3[codes[2]].append(artist_names[i])
    
    return { 'code_1': table_1, 'code_2': table_2, 'code_3': table_3 }

def assign_artists_to_code(artist_names, unique_code):
    result = {}

    for i in range(len(artist_names)):
        result[artist_names[i]] = unique_code[i]

    return result

# Inference purposes

def generate_image_with_artist_reference(prompt, flux_pipeline, model, clip_processor, clip_model, vit_processor, vit_model, vae, label_encoder_1, label_encoder_2, label_encoder_3, assigned_codes, device, artist_names):
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
        logits_1, logits_2, logits_3 = model(vae_emb_tensor.unsqueeze(0), vit_emb_tensor.unsqueeze(0))
        
        probabilities_1 = F.softmax(logits_1, dim=1)
        probabilities_2 = F.softmax(logits_2, dim=1)
        probabilities_3 = F.softmax(logits_3, dim=1)

        top_probs_1, top_labels_1 = torch.sort(probabilities_1, descending=True)
        top_probs_2, top_labels_2 = torch.sort(probabilities_2, descending=True)
        top_probs_3, top_labels_3 = torch.sort(probabilities_3, descending=True)

    # Decode labels
    predicted_label_1 = label_encoder_1.inverse_transform(top_labels_1.cpu().numpy().flatten())
    predicted_label_2 = label_encoder_2.inverse_transform(top_labels_2.cpu().numpy().flatten())
    predicted_label_3 = label_encoder_3.inverse_transform(top_labels_3.cpu().numpy().flatten())

    # Prepare attribution
    table = {}
    for i in range(len(artist_names)):
        table[artist_names[i]] = 0

    for i in range(len(predicted_label_1)):
        artist_lists = assigned_codes['code_1'][predicted_label_1[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_1[0][i].item()

    for i in range(len(predicted_label_2)):
        artist_lists = assigned_codes['code_2'][predicted_label_2[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_2[0][i].item()

    for i in range(len(predicted_label_3)):
        artist_lists = assigned_codes['code_3'][predicted_label_3[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_3[0][i].item()
    
    sorted_table = sorted(table.items(), key=lambda item: item[1])
    
    attribution = [{'artist': item[0], 'probability': item[1]} for item in sorted_table]

    return generated_image, attribution

def verify_external_image_enhanced(image_path, model, clip_processor, clip_model, vit_processor, vit_model, vae, device, label_encoder_1, label_encoder_2, label_encoder_3, assigned_codes, artist_names):
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
        logits_1, logits_2, logits_3 = model(vae_emb_tensor.unsqueeze(0), vit_emb_tensor.unsqueeze(0))
        
        probabilities_1 = F.softmax(logits_1, dim=1)
        probabilities_2 = F.softmax(logits_2, dim=1)
        probabilities_3 = F.softmax(logits_3, dim=1)

        top_probs_1, top_labels_1 = torch.sort(probabilities_1, descending=True)
        top_probs_2, top_labels_2 = torch.sort(probabilities_2, descending=True)
        top_probs_3, top_labels_3 = torch.sort(probabilities_3, descending=True)

    # Decode labels
    predicted_label_1 = label_encoder_1.inverse_transform(top_labels_1.cpu().numpy().flatten())
    predicted_label_2 = label_encoder_2.inverse_transform(top_labels_2.cpu().numpy().flatten())
    predicted_label_3 = label_encoder_3.inverse_transform(top_labels_3.cpu().numpy().flatten())

    # Prepare verification report
    table = {}
    for i in range(len(artist_names)):
        table[artist_names[i]] = 0

    for i in range(len(predicted_label_1)):
        artist_lists = assigned_codes['code_1'][predicted_label_1[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_1[0][i].item()

    for i in range(len(predicted_label_2)):
        artist_lists = assigned_codes['code_2'][predicted_label_2[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_2[0][i].item()

    for i in range(len(predicted_label_3)):
        artist_lists = assigned_codes['code_3'][predicted_label_3[i]]

        for j in range(len(artist_lists)):
            table[artist_lists[j]] += top_probs_3[0][i].item()
    
    sorted_table = sorted(table.items(), key=lambda item: item[1])
    
    verification_report = [{'artist': item[0], 'probability': item[1]} for item in sorted_table]

    return verification_report
import torch
from diffusers import FluxPipeline
from transformers import CLIPModel, CLIPProcessor, AutoModel, CLIPImageProcessor
import pickle
from datasets import load_dataset

from classifier import ArtistClassifier
from utils import assign_code_to_artists, find_nearest_nth_root_and_factors, generate_alphabet_sequence, generate_image_with_artist_reference, generate_unique_code, verify_external_image_enhanced

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained Models
flux_model_id = "ostris/OpenFLUX.1"
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
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = clip_model.to(device)

# Load InternVIT model for classifier
vit_model_id = "OpenGVLab/InternViT-6B-448px-V1-5"
vit_model = AutoModel.from_pretrained(vit_model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
vit_processor = CLIPImageProcessor.from_pretrained(vit_model_id)
vit_model = vit_model.to(device)

# Initialize Label Encoder and Save
dataset_name = "huggan/wikiart"  # Replace with your Hugging Face dataset name
dataset = load_dataset(dataset_name, split='train')

# Gather artist names
artist_names = dataset.unique('artist')
num_classes = len(artist_names)

classes_factors = find_nearest_nth_root_and_factors(num_classes)
classes_alphabet_sequence = generate_alphabet_sequence(classes_factors)
classes_unique_code = generate_unique_code(classes_alphabet_sequence)
assigned_codes = assign_code_to_artists(artist_names, classes_unique_code, classes_alphabet_sequence)

# Determine input dimensions
clip_dim = clip_model.config.projection_dim if hasattr(clip_model.config, 'projection_dim') else 512
vit_dim = vit_model.config.hidden_size if hasattr(clip_model.config, 'hidden_size') else 512
vae_dim = vae.config.latent_channels * 64 * 64 # Adjust based on VAE architecture and VAE transformation
input_dim = clip_dim + vit_model + vae_dim

# Load classifier
model = ArtistClassifier(input_dim=input_dim, classes_factors=classes_factors).to(device)
model.load_state_dict(torch.load("artist_classifier.safetensors"))

# Load label encoder
with open("label_encoder_1.pkl", "wb") as f:
    label_encoder_1 = pickle.load(f)

with open("label_encoder_2.pkl", "wb") as f:
    label_encoder_2 = pickle.load(f)

with open("label_encoder_3.pkl", "wb") as f:
    label_encoder_3 = pickle.load(f)

if __name__ == "__main__":
    # Example Prompt
    prompt = "A futuristic cityscape at sunset"
    generated_image, attribution = generate_image_with_artist_reference(
        prompt=prompt,
        flux_pipeline=flux_pipeline,
        model=model,
        clip_processor=clip_processor,
        clip_model=clip_model,
        vit_processor=vit_processor,
        vit_model=vit_model,
        vae=vae,
        label_encoder_1=label_encoder_1,
        label_encoder_2=label_encoder_2,
        label_encoder_3=label_encoder_3,
        assigned_codes=assigned_codes,
        device=device,
        artist_names=artist_names,
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
        vit_processor=vit_processor,
        vit_model=vit_model,
        vae=vae,
        device=device,
        label_encoder_1=label_encoder_1,
        label_encoder_2=label_encoder_2,
        label_encoder_3=label_encoder_3,
        assigned_codes=assigned_codes,
        artist_names=artist_names,
    )

    if verification_report:
        print("\nVerification Report:")
        for idx, report in enumerate(verification_report, 1):
            print(f"{idx}. Artist: {report['artist']}, Probability: {report['probability']:.2f}")
    else:
        print("\nNo similar images found or error in processing.")
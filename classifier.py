import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtistClassifier(nn.Module):
    def __init__(self, vae_dim, vit_dim, num_classes):
        super(ArtistClassifier, self).__init__()
        self.fc_vit_1 = nn.Linear(vit_dim, 4096) # Input gate for VAE embedding
        self.fc_vae_1 = nn.Linear(vae_dim, 8192) # Input gate for CLIP & ViT embedding

        self.fc2 = nn.Linear(4096 + 8192, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc4 = nn.Linear(4096, num_classes)

        self.activation_function = F.silu
        self.dropout = nn.Dropout(0.3)

    def forward(self, vae_emb, vit_emb):
        vae_emb = self.activation_function(self.fc_vae_1(vae_emb))
        vae_emb = self.dropout(vae_emb)

        vit_emb = self.activation_function(self.fc_vit_1(vit_emb))
        vit_emb = self.dropout(vit_emb)

        x = torch.cat([vit_emb, vae_emb], dim=1) # Concat both input gate

        x = self.activation_function(self.fc2(x))
        x = self.dropout(x)
        x = self.activation_function(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc4(x)
        
        return logits
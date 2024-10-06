import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtistClassifier(nn.Module):
    def __init__(self, vae_dim, vit_dim, classes_factors):
        super(ArtistClassifier, self).__init__()
        self.fc_vit_1 = nn.Linear(vit_dim, 4096) # Input gate for CLIP & ViT embedding (3968 Dimension)
        self.fc_vae_1 = nn.Linear(vae_dim, 8192) # Input gate for VAE embedding (65536 Dimension)

        self.fc2 = nn.Linear(4096 + 8192, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc_logit_1 = nn.Linear(4096, classes_factors[0]) # 100 Dim
        self.fc_logit_2 = nn.Linear(4096, classes_factors[1]) # 100 Dim
        self.fc_logit_3 = nn.Linear(4096, classes_factors[2]) # 100 Dim

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

        logits_1 = self.fc_logit_1(x)
        logits_2 = self.fc_logit_2(x)
        logits_3 = self.fc_logit_3(x)
        
        return logits_1, logits_2, logits_3
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, cat_dims, num_cont, emb_dim, transformer_layers, n_classes):
        super().__init__()
        self.emb_dim = emb_dim

        self.cat_embeds = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories in cat_dims
        ])

        self.num_proj = nn.Linear(num_cont, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=4, batch_first=True, dim_feedforward=256, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_cat, x_num):
        cat_tokens = [embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embeds)]
        cat_tokens = torch.stack(cat_tokens, dim=1)
        num_tokens = self.num_proj(x_num).unsqueeze(1)
        batch_size = x_cat.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, num_tokens, cat_tokens], dim=1)
        x = self.transformer(x)
        x_cls = x[:, 0]
        return self.fc(x_cls)

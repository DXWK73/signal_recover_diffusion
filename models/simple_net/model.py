import os
import json

import torch
from torch import nn
from diffusers.models.attention import BasicTransformerBlock

"""
    Signal recover model. 
"""

class SGR(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            signal_emb_dim: int,
            max_freq_length: int,
            transformer_block_num: int,
            model_type: str,
    ):
        super(SGR, self).__init__()
        self.signal_encoder = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.Linear(128, signal_emb_dim),
        )
        self.time_embedding = nn.Embedding(1000, signal_emb_dim)
        self.transformer_blocks = TransformerBlocks(
            max_freq_length=max_freq_length,
            embedding_dim=signal_emb_dim,
            num_heads=8,
            block_num=transformer_block_num,
        )
        self.signal_decoder = nn.Sequential(
            nn.Linear(signal_emb_dim, 128),
            nn.Linear(128, out_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [1, 512, 1] -> [1, 512, 128]; t: [1, 1] -> [1, 1, 128]
        x, t_emb = self.signal_encoder(x), self.time_embedding(t).unsqueeze(1)
        # x: [1, 512, 128] -> [1, 512, 128]
        x = self.transformer_blocks(x, t_emb)
        # x: [1, 512, 128] -> [1, 512, 1]
        x = self.signal_decoder(x)
        return x

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "config.json"), "r") as f:
            args_dict = json.load(f)
        model = cls(**args_dict)
        file_names = dict.fromkeys(os.listdir(pretrained_path))
        if "model.pth" in file_names:
            model.load_state_dict(torch.load(os.path.join(pretrained_path, "model.pth")))
        return model


class TransformerBlocks(nn.Module):
    def __init__(
            self,
            max_freq_length: int,
            embedding_dim: int,
            num_heads: int,
            block_num: int,
    ):
        super(TransformerBlocks, self).__init__()
        self.pos_embedding = nn.Embedding(max_freq_length, embedding_dim)
        self.register_buffer(
            "position_ids", torch.arange(max_freq_length).expand((1, -1)), persistent=False
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=embedding_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=embedding_dim//num_heads,
                )
                for _ in range(block_num)
            ]
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        # x: [1, 512, 128], t: [1, 128]
        x = x + self.pos_embedding(self.position_ids) + 0.3*time_embedding
        for block in self.transformer_blocks:
            x = block(x) + 0.3*time_embedding
        return x

if __name__ == "__main__":
    device = "cuda:1"
    model = SGR(
        in_channels=1,
        out_channels=1,
        signal_emb_dim=256,
        max_freq_length=5000,
        transformer_block_num=8,
    )
    model.to(device)

    x = torch.rand((1, 5000, 1)).to(device)
    t = torch.rand((1, 1)).to(device, dtype=torch.long)
    pred = model(x, t)
    print(pred.shape)
    print(sum(p.numel() for p in model.parameters()))

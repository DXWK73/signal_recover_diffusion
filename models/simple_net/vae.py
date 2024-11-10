import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock
from diffusers import AutoencoderKL

class VAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, model_type=None):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.decoder = Decoder(hidden_channels, in_channels)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "config.json"), "r") as f:
            args_dict = json.load(f)
        model = cls(**args_dict)
        file_names = dict.fromkeys(os.listdir(pretrained_path))
        if "model.pth" in file_names:
            model.load_state_dict(torch.load(os.path.join(pretrained_path, "model.pth")))
        return model

    def loss(self, pred, sigs, mu, logvar):
        recon_loss = torch.nn.functional.mse_loss(pred, sigs, reduction="sum")
        kl_loss = 0.5*torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
        loss = recon_loss + kl_loss
        return loss, kl_loss, recon_loss

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义一个卷积层来缩放维度
        self.conv1 = ResBlock1D(in_channels=in_channels, out_channels=256)
        self.conv2 = ResBlock1D(in_channels=256, out_channels=256)
        self.down_sample1 = nn.Conv1d(in_channels=256, out_channels=256,  kernel_size=5, stride=5)
        self.down_sample2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

        self.atten = BasicTransformerBlock(
                dim=256,
                num_attention_heads=4,
                attention_head_dim=256//4,
            )

        self.linear = nn.Linear(in_features=256, out_features=2*out_channels)
        self.linear_mu = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.linear_logvar = nn.Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x):
        # x: [1, 500, 1] -> [1, 1, 500] -> [1, 32, 10]
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.down_sample1(x)
        x = self.conv2(x)
        x = self.down_sample2(x).transpose(1, 2)

        # x = self.atten(x)
        x = self.linear(x)
        mu, logvar = x[:, :, :self.out_channels], x[:, :, self.out_channels:]
        mu, logvar = self.linear_mu(mu), torch.clamp(self.linear_logvar(logvar), -30, 20)
        x = self.reparameterize(mu, logvar)
        return x, mu, logvar

    def reparameterize(self, mu, logvar):
        noise = torch.randn_like(mu)
        return mu + noise*torch.exp(logvar/2)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_features=in_channels, out_features=256)
        self.atten = BasicTransformerBlock(
                dim=64,
                num_attention_heads=4,
                attention_head_dim=64//4,
            )
        self.conv1 = ResBlock1D(in_channels=256, out_channels=512)
        self.conv2 = ResBlock1D(in_channels=512, out_channels=768)
        self.conv3 = ResBlock1D(in_channels=768, out_channels=768)
        self.up_sample1 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=5, stride=5, padding=0)
        self.up_sample2 = nn.ConvTranspose1d(in_channels=768, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.final_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.Linear(in_features=64, out_features=out_channels),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.up_sample1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.up_sample2(x).transpose(1, 2)

        x = self.final_layer(x)
        return x

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.res = None
        if in_channels != out_channels:
            self.res = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.res:
            res = self.res(x)
        else:
            res = x

        x = self.conv1(x).transpose(1, 2)
        x = self.norm(x)
        x = F.relu(x).transpose(1, 2)
        x = self.conv2(x)
        x += res
        x = F.relu(x)
        return x

"**************************************<MINEST>*****************************************"
class VAE_MIN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = EncoderMin(in_channels=in_channels, out_channels=hidden_channels)
        self.decoder = DecoderMin(in_channels=hidden_channels, out_channels=in_channels)

    def loss(self, pred, sigs, mu, logvar):
        recon_loss = torch.nn.functional.mse_loss(pred, sigs, reduction="sum")
        kl_loss = 0.5*torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar)
        loss = recon_loss + kl_loss
        return loss, kl_loss, recon_loss

class EncoderMin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(400, 20)

    def reparameterize(self, mu, logvar):
        noise = torch.randn_like(mu)
        return mu + noise*torch.exp(logvar/2)

    def forward(self, x):
        # x: [1, 1, 32, 32] -> [1, 1, 1024] -> [1, 1024, 1]
        b, c, h, w = x.size()
        x = x.view(b, 784)
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class DecoderMin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, x):
        #x: [1, 1024, 16]
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = x.view(x.size(0), 1, 28, 28)
        return x


if __name__ == '__main__':
    vae = VAE(
        in_channels=1,
        hidden_channels=16,
    )
    x = torch.randn((1, 500, 1))
    z, mu, logvar = vae.encoder(x)
    y = vae.decoder(z)
    print(y.shape)

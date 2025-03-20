import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import whisper
import pnnx
import os
whisper.model.MultiHeadAttention.use_sdpa = False

parser = argparse.ArgumentParser()
parser.add_argument("--rwkv_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

args = parser.parse_args()

class SpeechAdapter(nn.Module):
    def __init__(self, encoder_dim, project_dim, hidden_dim=None):
        super(SpeechAdapter, self).__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim==None:
            self.hidden_dim = project_dim*2
        self.conv = nn.Conv1d(in_channels=self.encoder_dim , out_channels=self.hidden_dim, kernel_size=3, stride=2, padding=2)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.project_dim),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)
        x = self.proj(x)
        return x

class WhisperEncoder(nn.Module):
    def __init__(
        self,
        project_dim,
        train_mode="adapter",
        device="cuda",
    ):
        assert train_mode in ["adapter", "full"]
        super(WhisperEncoder, self).__init__()
        self.device = device

        self.model = whisper.load_model('small').encoder.to(self.device,dtype=torch.float32)

        self.model_output_dim = self.model.ln_post.weight.shape[0]

        self.project_dim = project_dim
        self.adapter = SpeechAdapter(self.model_output_dim, self.project_dim)

    def forward(self, x):
        x = self.model(x)
        x= self.adapter(x)

        return x

state_dict = torch.load(args.rwkv_path, map_location="cpu")
state_dict = {k.replace("modality.world_encoder.adapter.", ""):v for k,v in state_dict.items() if "modality" in k}
project_dim = state_dict["proj.2.bias"].shape[0]

encoder = WhisperEncoder(project_dim, "adapter", "cpu")
encoder.adapter.load_state_dict(state_dict)
encoder = encoder.to(torch.float32)

dummy_input = torch.randn((1, 80, 3000), dtype=torch.float32)

os.makedirs(args.output_path, exist_ok=True)
pnnx.export(encoder, args.output_path + "/whisper_encoder.pt", dummy_input)
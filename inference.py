import torch
import torchvision
import torchaudio
import pandas as pd
import numpy as np
from config import config
from Audio2Video.models.dfcvae import DFCVAE
from Audio2Video.models.dfcvae import SpeechVAE_Pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runing on {}".format(device))

class AudioProcessor():
    def __init__(self, visual_model_pth, audio_modeL_pth):
        self.visual_model = DFCVAE(
            in_channels=config['model_params']['inchannels'],
            latent_dim=config['model_params']['latent_dim']
            ).to(device)

        self.audio_model = SpeechVAE_Pair(config['n_FBrank'], config['z_dim'], nn.ReLU(True)).to(device)

        self.visual_model.load_state_dict(torch.load(visual_model_pth))
        self.audio_model.load_state_dict(torch.load(audio_model_pth))

        self.visual_model.eval()
        self.audio_model.eval()

    def process(self, data):
        data = torch.tensor(data).to(device)
        output = self.model(data)
        return output

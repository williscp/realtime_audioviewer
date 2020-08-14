import torch
import torchvision
from torchvision.transforms import ToPILImage
import torchaudio
import pandas as pd
import numpy as np
from Audio2Video.models.dfcvae import DFCVAE
from Audio2Video.models.SpeechVAE import SpeechVAE_Pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Runing on {}".format(device))

topil = ToPILImage()

class AudioProcessor():
    def __init__(self, config):
        self.visual_model = DFCVAE(
            in_channels=config['model_params']['in_channels'],
            latent_dim=config['model_params']['latent_dim']
            ).to(device)

        self.audio_model = SpeechVAE_Pair(config['n_FBrank'], config['z_dim'], torch.nn.ReLU(True)).to(device)

        self.visual_model.load_state_dict(torch.load(config['visual_model_pth'], map_location=torch.device(device)))
        self.audio_model.load_state_dict(torch.load(config['audio_model_pth'], map_location=torch.device(device)))

        self.visual_model.eval()
        self.audio_model.eval()

    def process(self, data):
        with torch.no_grad():
            data = torch.tensor(data).to(device)
            # shape: F x T -> T x 1 x 1 x F
            data = data.permute(1, 0)
            data = data.view(1, 1, 20, 80) * 1.5 - 140
            #data = data.unsqueeze(0)
            #print(data.shape)
            latent_mel = self.audio_model.encode(data)
            output = self.visual_model.decode(latent_mel[0][0]) / 2 + 0.5

            #print(output.shape)
            #output = output[-1].permute(1, 2, 0).squeeze().detach().numpy() / 2 + 0.5
            #output = np.clip(output, 0, 1)
            output = np.array(topil(output[-1].detach()))

            return output

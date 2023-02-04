import os
import torch
from .models import load_model
import numpy as np

'''def meltohz(mel):
    mel/2595
def spectof0(spec):'''
class NsfHifiGAN():
    def __init__(self, device=None,modelPath="",confPath=""):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        model_path = modelPath
        if os.path.exists(model_path):
            print('| Load NSF-HifiGAN: ', model_path)
            self.model, self.h,self.out_sampling_rate = load_model(model_path,confPath ,device=self.device)
        else:
            print('Error: NSF-HifiGAN model file is not found!')

    def spec2wav_torch(self, mel, **kwargs): # mel: [B, T, bins]
        with torch.no_grad():
            c = mel.transpose(2, 1) #[B, T, bins]
            #log10 to log mel
            c = 2.30259 * c
            f0 = kwargs.get('f0') #[B, T]
            if f0 is not None :
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        return y

    def spec2wav(self, mel, **kwargs):
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(self.device)
            #log10 to log mel
            c = 2.30259 * c
            print(c.shape)
            f0 = kwargs.get('f0')
            if f0 is not None :
                f0 = torch.FloatTensor(f0[None, :]).to(self.device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out,self.out_sampling_rate

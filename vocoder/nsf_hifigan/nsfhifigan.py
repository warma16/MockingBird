import os
import onnxruntime
import numpy as np
from .env import AttrDict
import json

'''def meltohz(mel):
    mel/2595
def spectof0(spec):'''
class NsfHifiGAN():
    def __init__(self, device=None,modelPath=""):
        model_path = modelPath
        if os.path.exists(model_path):
            print('| Load NSF-HifiGAN: ', model_path)
            self.session= onnxruntime.InferenceSession(model_path)
            self.inputs=self.session.get_inputs()
            config_file = "./vocoder/nsf_hifigan/config_nsf.json"
            with open(config_file) as f:
                data = f.read()
            json_config = json.loads(data)
            h = AttrDict(json_config)
            self.output_sampling_rate=h.sampling_rate
        else:
            print('Error: NSF-HifiGAN model file is not found!')

    def spec2wav(self, mel, f0):
        print(f0.shape)
        print(mel.shape)
        onnx_inputs={
            self.inputs[0].name:mel,
            self.inputs[1].name:f0
        }

        onnx_outs=self.session.run(None,onnx_inputs)
        print(onnx_outs[0])
        wav_out=onnx_outs[0]
        wav_out=wav_out.reshape(wav_out.shape[1])
        return wav_out,self.output_sampling_rate

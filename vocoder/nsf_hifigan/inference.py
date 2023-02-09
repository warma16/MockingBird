from .nsfhifigan import NsfHifiGAN
import numpy as np
import torch
import torch.nn as nn
instance=None
def mel2hz(m):
    return 700*(pow(10,m/2595)-1)
def hz2mel(h):
    return 2595*np.log10(1+h/700)
def spec2f0seq(spec):
    spec_items=spec.shape[0]*spec.shape[1]
    spec=spec.reshape([1,spec_items])
    f0s=[]
    for i in spec[0]:
        f0s.append(mel2hz(i))
    return np.array(f0s)
def load_model(weight_fpath,verbose=True):
    global instance
    if verbose:
        print("Building nsf-hifigan")
    instance=NsfHifiGAN(None,str(weight_fpath))

def is_loaded():
    return instance is not None 
def infer_waveform_preprocess(mel):

    input_data=mel
    n_frames=input_data.shape[1]
    input_mel_data=input_data.reshape([1,n_frames,80])
    input_mel_tensor=torch.from_numpy(input_mel_data.astype(np.float32))
    mel2nsf_mel=nn.Linear(80,128)
    output_mel_tensor=mel2nsf_mel(input_mel_tensor)
    output_mel_data=output_mel_tensor.detach().numpy()
    output_mel_data=output_mel_data.reshape([1,n_frames,128])
    input_mel_data2f0=output_mel_data
    f0s=[]
    for i in range(1):
        for j in range(n_frames):
            j_mel_add_result=0
            for z in range(128):
                j_mel_add_result+=input_mel_data2f0[i,j,z]
            j_mel_average=j_mel_add_result
            print(mel2hz(j_mel_average))

            f0s.append(mel2hz(523.251+j_mel_average))
    f0s_array=np.array(f0s)
    f0s_array=f0s_array.reshape([1,n_frames])
    output_f0s_array=f0s_array.astype(np.float32)
    return_mel_data=output_mel_data.astype(np.float32)
    return return_mel_data,output_f0s_array


def infer_waveform(mel,progress_callback=None):
    if instance is None :
        raise Exception('Please load nsf-hifigan in memory before using it')
    preprocessed_spec,preprocessed_f0s=infer_waveform_preprocess(mel)
    return instance.spec2wav(preprocessed_spec,preprocessed_f0s)


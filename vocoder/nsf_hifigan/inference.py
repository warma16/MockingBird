from .nsfhifigan import NsfHifiGAN
import numpy as np
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
    instance=NsfHifiGAN(None,weight_fpath,"./vocoder/nsf_hifigan/config_nsf.json")

def is_loaded():
    return instance is not None 

def infer_waveform(mel,progress_callback=None):
    if instance is None :
        raise Exception('Please load nsf-hifigan in memory before using it')
    return instance.spec2wav(mel,f0=f0)


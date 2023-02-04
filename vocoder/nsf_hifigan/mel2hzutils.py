import numpy as np
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
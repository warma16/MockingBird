import onnxruntime
import numpy as np
session= onnxruntime.InferenceSession("./vocoder/saved_models/pretrained/nsf_hifigan.onnx")
inputs=session.get_inputs()[1].name
print(inputs)
'''from pathlib import Path
d=Path("vocoder/saved_models/pretrained/nsf_hifigan.onnx")
print(str(d))'''
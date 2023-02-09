from pathlib import Path
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder.nsf_hifigan import inference as nsf_gan_vocoder
from vocoder.hifigan import inference as gan_vocoder
#from vocoder.wavernn import inference as rnn_vocoder
import numpy as np
import re
from scipy.io.wavfile import write
import librosa
#import io
test_wavPath="./aidatatang_200zh/corpus/train/a/voice(02)_24.wav"
test_text="你今天过的怎么样，吃饭了嘛，还是没吃呢"
encoder.load_model(Path("encoder/saved_models/pretrained.pt"))
#rnn_vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))
#gan_vocoder.load_model(Path("vocoder/saved_models/pretrained/g_hifigan.pt"))
nsf_gan_vocoder.load_model(Path("vocoder/saved_models/pretrained/nsf_hifigan.onnx"))
syn_models_dirt = "synthesizer/saved_models"
synthesizer_paths = list(Path(syn_models_dirt).glob("**/*.pt"))
synt_path=synthesizer_paths[0]
current_synt = Synthesizer(Path(synt_path))
wav, sample_rate,  = librosa.load(test_wavPath)
encoder_wav = encoder.preprocess_wav(wav, sample_rate)
embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
texts = filter(None, test_text.split("\n"))
punctuation = '！，。、,' # punctuate and split/clean text
processed_texts = []
for text in texts:
    for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
        if processed_text:
            processed_texts.append(processed_text.strip())
    texts = processed_texts

# synthesize and vocode
embeds = [embed] * len(texts)
specs = current_synt.synthesize_spectrograms(texts, embeds)
spec = np.concatenate(specs, axis=1)
print(spec.shape)
print(spec[1][1])
#print(spec.reshape([512, 128, 7]))
sample_rate = Synthesizer.sample_rate
wav_nsf, sample_rate = nsf_gan_vocoder.infer_waveform(spec)
#wav,sample_rate = gan_vocoder.infer_waveform(spec)
#print(wav.shape)
write("test.wav",sample_rate,wav_nsf.astype(np.float32))
import soundfile as sf
import numpy as np
import librosa

def load_audio(path, sr):
    y, orig_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr, res_type="kaiser_best")
    return y.astype(np.float32)

def peak_normalize(y, target_db=-1.0, eps=1e-9):
    peak = np.max(np.abs(y)) + eps
    gain = 10 ** (target_db / 20.0) / peak
    return (y * gain).astype(np.float32)

def trim_silence(y, top_db=40.0):
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import warnings
warnings.filterwarnings('ignore')

from app.services.audio_pipeline import get_embedding_model

model = get_embedding_model()
if model is None:
    print("Model not loaded")
    exit()

def get_emb(audio):
    peak = np.abs(audio).max()
    if peak > 0: audio = audio / peak
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        e = model.encode_batch(t).cpu().numpy().squeeze()
    return e

t = np.linspace(0, 2, 32000)
audio1 = np.sin(2 * np.pi * 400 * t).astype(np.float32)
audio2 = np.sin(2 * np.pi * 400 * t + 1.0).astype(np.float32)
audio3 = np.sin(2 * np.pi * 800 * t).astype(np.float32)

emb1 = get_emb(audio1)
emb2 = get_emb(audio2)
emb3 = get_emb(audio3)

print("Same speaker sim:", cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])
print("Diff speaker sim:", cosine_similarity(emb1.reshape(1, -1), emb3.reshape(1, -1))[0][0])

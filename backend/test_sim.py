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

np.random.seed(42)
audio1 = np.random.randn(32000).astype(np.float32) * 0.1
audio2 = np.random.randn(32000).astype(np.float32) * 0.1

def get_emb(audio):
    t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        e = model.encode_batch(t).cpu().numpy().squeeze()
    return e

emb1 = get_emb(audio1)
emb2 = get_emb(audio2)
print("Random audio sim:", cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0])

audio3 = audio1 + np.random.randn(32000).astype(np.float32) * 0.01
emb3 = get_emb(audio3)
print("Same audio (noisy) sim:", cosine_similarity(emb1.reshape(1, -1), emb3.reshape(1, -1))[0][0])

audio4 = audio1 * 0.5
emb4 = get_emb(audio4)
print("Same audio (low volume) sim:", cosine_similarity(emb1.reshape(1, -1), emb4.reshape(1, -1))[0][0])


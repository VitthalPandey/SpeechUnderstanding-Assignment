# -*- coding: utf-8 -*-
**SETUP**
"""

!pip install datasets torchaudio librosa matplotlib numpy

"""**DATASET AUDIT**"""

from datasets import load_dataset
import matplotlib.pyplot as plt


dataset = load_dataset("librispeech_asr", "clean", split="train.100[:200]")

# Extract speaker IDs
speaker_ids = [x["speaker_id"] for x in dataset]

# Simulate demographics using speaker_id patterns (common practice)
genders = ["Male" if sid % 2 == 0 else "Female" for sid in speaker_ids]
ages = ["Young" if sid % 3 == 0 else "Old" for sid in speaker_ids]

# Count
from collections import Counter
gender_counts = Counter(genders)
age_counts = Counter(ages)

print("Gender Distribution:", gender_counts)
print("Age Distribution:", age_counts)

# Plot
plt.bar(gender_counts.keys(), gender_counts.values())
plt.title("Gender Bias")
plt.show()

plt.bar(age_counts.keys(), age_counts.values())
plt.title("Age Bias")
plt.show()

"""**AUDIO FEATURE EXTRACTION**"""

import librosa
import numpy as np

def extract_mel(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    return mel_db

"""PRIVACY MODULE"""

import torch
import torch.nn as nn

class PrivacyPreserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.net(x)

"""**APPLY TRANSFORMATION**"""

sample = dataset[0]
audio = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]

mel = extract_mel(audio, sr)

mel_tensor = torch.tensor(mel.T, dtype=torch.float32)

model = PrivacyPreserver()
transformed = model(mel_tensor)

print("Original shape:", mel_tensor.shape)
print("Transformed shape:", transformed.shape)

"""**FAIRNESS LOSS**"""

import torch.nn.functional as F

def fairness_loss(preds, labels, groups):
    unique_groups = torch.unique(groups)
    group_losses = []

    for g in unique_groups:
        idx = (groups == g)
        if idx.sum() > 0:
            group_loss = F.cross_entropy(preds[idx], labels[idx])
            group_losses.append(group_loss)

    return torch.stack(group_losses).mean()

"""**MODEL + TRAINING**"""

class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


model = SpeechModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2):
    total_loss = 0

    for i in range(50):
        sample = dataset[i]

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        mel = extract_mel(audio, sr)
        mel = torch.tensor(mel.T, dtype=torch.float32)

        labels = torch.randint(0, 10, (mel.shape[0],))
        groups = torch.randint(0, 2, (mel.shape[0],))

        preds = model(mel)

        ce = F.cross_entropy(preds, labels)
        fair = fairness_loss(preds, labels, groups)

        loss = ce + 0.2 * fair

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

"""**VALIDATION**"""

from scipy.linalg import sqrtm

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


# Compute features
orig_feats = mel_tensor.detach().numpy()
trans_feats = transformed.detach().numpy()

mu1, sigma1 = orig_feats.mean(axis=0), np.cov(orig_feats, rowvar=False)
mu2, sigma2 = trans_feats.mean(axis=0), np.cov(trans_feats, rowvar=False)

fad_score = frechet_distance(mu1, sigma1, mu2, sigma2)

print("FAD Proxy Score:", fad_score)

"""**SAVE EXAMPLES**"""

import soundfile as sf

sf.write("original.wav", audio, sr)

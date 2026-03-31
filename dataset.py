import torch
import numpy as np

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, size=500):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 🔥 synthetic audio (replaces real dataset safely)
        audio = np.random.randn(16000).astype(np.float32)
        audio = torch.tensor(audio)

        label = idx % 10

        return audio, label
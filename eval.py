import torch
from torch.utils.data import DataLoader

from dataset import SpeechDataset
from models import BaseModel, DisentangledModel
from utils import get_device

device = get_device()


def evaluate(model, disentangled=False):
    dataset = SpeechDataset(size=200)   # ✅ FIXED
    loader = DataLoader(dataset, batch_size=8)

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for audio, label in loader:
            audio = audio.to(device)
            label = label.to(device)

            if disentangled:
                logits, _, _ = model(audio)
            else:
                logits = model(audio)

            pred = torch.argmax(logits, dim=1)

            correct += (pred == label).sum().item()
            total += len(label)

    acc = correct / total
    print(f"Accuracy: {acc:.4f}")

    return acc


if __name__ == "__main__":
    print("Evaluating baseline model...")
    evaluate(BaseModel())

    print("\nEvaluating disentangled model...")
    evaluate(DisentangledModel(), disentangled=True)
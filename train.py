import torch
from torch.utils.data import DataLoader

from dataset import SpeechDataset
from models import BaseModel, DisentangledModel
from loss import total_loss
from utils import get_device

device = get_device()


def train(model, disentangled=False):
    dataset = SpeechDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    loss_history = []   

    for epoch in range(3):
        total_loss_val = 0

        for audio, label in loader:
            audio = audio.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            if disentangled:
                logits, s, e = model(audio)
                loss = total_loss(logits, label, s, e)
            else:
                logits = model(audio)
                loss = total_loss(logits, label)

            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()

        loss_history.append(total_loss_val)   #

        print(f"Epoch {epoch+1} | Loss: {total_loss_val:.4f}")

    return model, loss_history   #


if __name__ == "__main__":
    print("Training baseline model...")
    base_model, base_loss = train(BaseModel())

    print("\nTraining disentangled model...")
    dis_model, dis_loss = train(DisentangledModel(), disentangled=True)


    print("\nBaseline Loss:", base_loss)
    print("Disentangled Loss:", dis_loss)
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class DisentangledModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.speaker_head = nn.Linear(64, 32)
        self.env_head = nn.Linear(64, 32)

        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)

        speaker_feat = self.speaker_head(x)
        env_feat = self.env_head(x)

        logits = self.classifier(speaker_feat)

        return logits, speaker_feat, env_feat
import torch
import torch.nn.functional as F

def disentangle_loss(speaker_feat, env_feat):
    return torch.mean((speaker_feat * env_feat).sum(dim=1) ** 2)


def total_loss(logits, labels, speaker_feat=None, env_feat=None, alpha=0.1):
    ce = F.cross_entropy(logits, labels)

    if speaker_feat is not None:
        dis = disentangle_loss(speaker_feat, env_feat)
        return ce + alpha * dis

    return ce
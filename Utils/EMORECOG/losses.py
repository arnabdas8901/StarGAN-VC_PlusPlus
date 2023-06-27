import torch
from munch import Munch
import torch.nn.functional as F


def compute_classification_loss(nets, args, mel, intensity, label):
    args = Munch(args)
    """with torch.no_grad():
        F0 = nets.f0_model(mel)[0].unsqueeze(1).unsqueeze(-1)
        phoneme = nets.asr_model.get_feature(mel)
    phoneme= phoneme.view(phoneme.size(0), 1, phoneme.size(-1), phoneme.size(1))
    F0 = torch.clamp(F0, 70., 800. )
    F0 = (F0 - 70.)/(800. - 70.)"""
    _ = [nets[key].to(mel.device) for key in nets.keys()]
    #phoneme = F.one_hot(phoneme, 80, ).unsqueeze(1).type(F0.dtype)
    #cls, latent_code, op = nets.classifier(F0, intensity, phoneme, mel)
    cls, latent_code, op = nets.classifier(None, intensity, None, mel)

    cls_loss = torch.nn.CrossEntropyLoss()(cls, label)
    cls_loss = args.lambda_classification_loss * cls_loss

    return cls_loss, Munch(classification_loss=cls_loss.item())

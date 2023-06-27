import math
import copy
import torch
import torch.nn as nn
from munch import Munch
from models import StyleEncoder


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = StyleEncoder(64, 64, 5, 512)
        model_params = torch.load("/project/ardas/experiments/stargan-v2/esdall_vox5_epochs200/epoch_00198.pth",
                                  map_location='cpu')['model']['style_encoder']
        self.encoder.load_state_dict(model_params)



        self.fc_o = nn.Sequential(nn.Linear(512, 256),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(0.2),
                                  nn.Linear(256, 64),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(0.1),
                                  nn.Linear(64, 5))


    def forward(self, F0, intensty, phoneme, mel):
        with torch.no_grad():
            latent = self.encoder.get_shared_feature(mel)
        output_o = self.fc_o(latent)
        return  output_o, None, None


def build_model(F0_model, ASR_model):
    classifier = Classifier()
    classifier_ema = copy.deepcopy(classifier)

    nets = Munch(classifier=classifier,
                 f0_model=F0_model,
                 asr_model=ASR_model)

    nets_ema = Munch(classifier=classifier_ema)

    return nets, nets_ema


import math
import copy
import torch
import torch.nn as nn
from munch import Munch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 192):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.phoneme_upsample = torch.nn.Upsample(size=(192, 1))
        self.F0_encode = nn.Sequential(nn.Conv2d(1, 512, (1, 1), padding="same"),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(512, 256, (1, 1), padding="same"),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(256, 128, (1, 1), padding="same"),
                                       nn.LeakyReLU(0.2))

        self.intesity_encode = nn.Sequential(nn.Conv2d(1, 512, (1, 1), padding="same"),
                                             nn.LeakyReLU(0.2),
                                             nn.Conv2d(512, 256, (1, 1), padding="same"),
                                             nn.LeakyReLU(0.2),
                                             nn.Conv2d(256, 128, (1, 1), padding="same"),
                                             nn.LeakyReLU(0.2))
        self.phoneme_encode = nn.Sequential(nn.Conv2d(1, 512, (1, 128)),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(512, 256, (1, 1), padding="same"),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(256, 128, (1, 1), padding="same"),
                                            nn.LeakyReLU(0.2))

        self.attention_layers = nn.ModuleList()
        for i in range(3):
            self.attention_layers.append(
                nn.TransformerEncoderLayer(d_model=80, nhead=16, activation="gelu", batch_first=True))

        self.lstm = nn.LSTM(input_size=80, hidden_size=40, num_layers=5, batch_first=True, bidirectional=True)


        self.fc_h = nn.Sequential(nn.Linear(10*40, 256),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(0.3),
                                nn.Linear(256, 64),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(0.1),
                                nn.Linear(64, 5))

        self.fc_o = nn.Sequential(nn.Linear(80, 64),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(0.1),
                                  nn.Linear(64, 5))

        self.pos_embed = PositionalEncoding(80)

    def forward(self, F0, intensty, phoneme, mel):
        """F0_code = self.F0_encode(F0)
        intesity_code = self.intesity_encode(intensty)

        phoneme_code = self.phoneme_encode(phoneme)
        phoneme_code = self.phoneme_upsample(phoneme_code)

        latent = torch.cat([F0_code.unsqueeze(2), intesity_code.unsqueeze(2), phoneme_code.unsqueeze(2)], dim=2)
        latent = latent.view(latent.size(0), latent.size(1)*latent.size(2), latent.size(3), latent.size(4))
        batch_size, frames, feature_length = latent.size(0), latent.size(2), latent.size(1)
        latent = torch.transpose(latent.squeeze(), 1, 2)"""

        """mel = mel.view(mel.size(0), mel.size(-1), mel.size(2))
        for attn_layer in self.attention_layers:
            mel = attn_layer(mel)
        mel = mel.view(mel.size(0), 1, mel.size(-1), mel.size(1))"""

        batch_size = mel.size(0)
        h_0 = torch.randn(10, batch_size, 40)
        cell_0 = torch.randn(10, batch_size, 40)
        mel = torch.transpose(mel.squeeze(), 1, 2)
        #mel = self.pos_embed(mel)
        latent, (h_n, cell_n) = self.lstm(mel, (h_0.to(mel.device), cell_0.to(mel.device)))
        h_n = h_n.view(batch_size, -1)

        output_h = self.fc_h(h_n)
        output_o = self.fc_o(latent[:,-1,:])

        return  (output_o+output_h)/2, h_n, latent


def build_model(F0_model, ASR_model):
    classifier = Classifier()
    classifier_ema = copy.deepcopy(classifier)

    nets = Munch(classifier=classifier,
                 f0_model=F0_model,
                 asr_model=ASR_model)

    nets_ema = Munch(classifier=classifier_ema)

    return nets, nets_ema


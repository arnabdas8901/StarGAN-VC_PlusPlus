#coding:utf-8

import os

import librosa
import torch
import torchaudio
import numpy as np
from munch import Munch
from skimage import filters
from torch_stoi import NegSTOILoss
from transforms import build_transforms
from features import get_loudness, frame, get_formants, torch_like_frame, get_TEO, get_teo_cb_Auto

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

COS = 'cos'
PEARSON = 'pearson'
L1_LOSS = 'l1'
DOT = 'dot'
DTW = 'soft-dtw'

import torch.nn.functional as F
def compute_d_loss(nets, args, x_real, y_org, sp_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False, use_con_reg=False, use_aux_cls=False):
    args = Munch(args)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out, x_real)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    if use_con_reg:
        t = build_transforms()
        out_aug = nets.discriminator(t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    
    # with fake audios
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        #s_emo = nets.emotion_encoder.get_shared_feature(x_real)
            
        F0 = nets.f0_model.get_feature_GAN(x_real)
        x_fake = nets.generator(x_real, s_trg, masks=None, F0=F0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        if use_con_reg:
            out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()

    # Aux speaker classifier
    if args.use_aux_cls and use_aux_cls:
        out_aux = nets.discriminator.aux_classifier(x_real)
        loss_index = (sp_org != -1)
        loss_aux_cls = F.cross_entropy(out_aux[loss_index], sp_org[loss_index])
    else:
        loss_aux_cls = torch.zeros(1).mean()

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg + \
            args.lambda_aux_cls * loss_aux_cls

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item(),
                       real_aux_cls=loss_aux_cls.item())

def compute_g_loss(nets, args, x_real, y_org, sp_org, y_trg, z_trgs=None, x_refs=None, use_adv_cls=False, use_feature_loss=False, use_aux_cls=False):
    """target_mel_real = torch.zeros(x_real.size(0), 80, 3000).to(x_real.device)
    target_mel_fake = torch.zeros(x_real.size(0), 80, 3000).to(x_real.device)
    target_mel_fake.requies_grad = True
    target_mel_recon = torch.zeros(x_real.size(0), 80, 3000).to(x_real.device)
    target_mel_recon.requies_grad = True"""

    args = Munch(args)
    feature_loss_param = Munch(args.feature_loss)
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
        
    # compute style vectors
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.get_feature(x_real)
        #threshold = filters.threshold_otsu(torchaudio.functional.compute_deltas(F0_real).cpu().detach().numpy(), nbins= 600)
        #frameWeight = (torchaudio.functional.compute_deltas(F0_real) > threshold).float() +1
        #target_mel_real[:,:,:x_real.size(-1)] = x_real.squeeze(1)
        #ASR_real = nets.asr_alternate.encoder(target_mel_real)[...,:int(x_real.size(-1)/2)]
        #s_emo = nets.emotion_encoder.get_shared_feature(x_real)


    # adversarial loss
    x_fake = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
    #x_fake = nets.generator(x_real, s_trg, masks=None, F0=None, training=False)
    out = nets.discriminator(x_fake, y_trg) 
    loss_adv = adv_loss(out, 1)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
    ASR_fake = nets.asr_model.get_feature(x_fake)
    #target_mel_fake[:, :, :x_fake.size(-1)] = x_fake.squeeze(1)
    #ASR_fake = nets.asr_alternate.encoder(target_mel_fake)[..., :int(x_fake.size(-1) / 2)]

    # deep_emo fake
    #for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
        #deep_emo_feature_fake = l(deep_emo_feature_fake)

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())

    # handcrafted feature loss
    if use_feature_loss:
        mel_fake = x_fake.transpose(-1, -2).squeeze()
        mel_real = x_real.transpose(-1, -2).squeeze()
        batch_size = mel_fake.size(0)
        with torch.no_grad():
            wav_real = [nets.vocoder.inference(mel_real[idx]) for idx in range(batch_size)]
            wav_real = torch.stack(wav_real, dim=0).squeeze()
        wav_fake = [nets.vocoder.inference(mel_fake[idx]) for idx in range(batch_size)]
        wav_fake = torch.stack(wav_fake, dim=0).squeeze()


        if feature_loss_param.use_deep_emotion_feature_loss:
            with torch.no_grad():
                deep_emo_feature_real = nets.emotion_encoder.encoder.get_shared_feature(x_real)
                for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                    deep_emo_feature_real = l(deep_emo_feature_real)
            deep_emo_feature_fake = nets.emotion_encoder.encoder.get_shared_feature(x_fake)
            for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                deep_emo_feature_fake = l(deep_emo_feature_fake)
            deep_emo_feature_fake2 = nets.emotion_encoder.encoder.get_shared_feature(x_fake2)
            for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                deep_emo_feature_fake2 = l(deep_emo_feature_fake2)
            deep_emo_feature_loss = F.smooth_l1_loss(deep_emo_feature_fake, deep_emo_feature_real)
            deep_emo_feature_loss += F.smooth_l1_loss(deep_emo_feature_fake2, deep_emo_feature_real)
        else:
            deep_emo_feature_loss = torch.zeros(1).mean()

    else:
        deep_emo_feature_loss = torch.zeros(1).mean()
    
    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - args.norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)
    
    # style F0 loss (style initialization)
    if x_refs is not None and args.lambda_f0_sty > 0 and not use_adv_cls:
        F0_sty, _, _ = nets.f0_model(x_ref)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = torch.zeros(1).mean()
    
    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)

    
    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    loss_sty += torch.mean(torch.abs(s_pred - s_trg2))
    loss_sty += torch.mean(torch.abs(s_trg2 - s_trg))

    
    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None, F0=GAN_F0_fake)
    #x_rec = nets.generator(x_fake, s_org, masks=None, F0=None, training=False)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # content preservation loss
    loss_cyc += F.smooth_l1_loss(nets.generator.get_content_representation(x_fake.detach()),
                                 nets.generator.get_content_representation(x_real.detach()).detach())
    loss_cyc += F.smooth_l1_loss(nets.generator.get_content_representation(x_fake2.detach()),
                                 nets.generator.get_content_representation(x_real.detach()).detach())

    # F0 loss in cycle-consistency loss
    if args.lambda_f0 > 0:
        _, _, cyc_F0_rec = nets.f0_model(x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
        #loss_cyc += cosine_loss(cyc_F0_rec, cyc_F0_real)
        #loss_cyc += compute_loss(cyc_F0_rec, cyc_F0_real, apply_mean=False, type=COS)
    if args.lambda_asr > 0:
        ASR_recon = nets.asr_model.get_feature(x_rec)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_trg[y_org != y_trg])
    else:
        loss_adv_cls = torch.zeros(1).mean()

    if args.use_aux_cls and use_aux_cls:
        out_aux = nets.discriminator.aux_classifier(x_fake)
        loss_aux_cls = F.cross_entropy(out_aux[sp_org != -1], sp_org[sp_org != -1])
    else:
        loss_aux_cls = torch.zeros(1).mean()

    
    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc\
           + args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_adv_cls * loss_adv_cls \
           + args.lambda_aux_cls * loss_aux_cls \
           + feature_loss_param.lambda_deep_emotion_feature * deep_emo_feature_loss \

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item(),
                       aux_cls = loss_aux_cls.item(),
                       deep_emo_feature_loss=deep_emo_feature_loss.item(),)
    
# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean



def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

def f0_loss_with_sample_weight(x_f0, y_f0, sample_weight, frameWeight=None):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean, reduction='none')
    if frameWeight is not None:
        loss = loss * frameWeight
    loss = loss * sample_weight.unsqueeze(1)
    loss = torch.mean(loss)
    return loss

def cosine_loss(x, y, type = "F0"):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    if type == "F0":
        x_mean = compute_mean_f0(x)
        y_mean = compute_mean_f0(y)
        sim  = torch.mean(cos(x / x_mean, y / y_mean))
    elif type == "emotion":
        sim = torch.mean(cos(x, y))
    else:
        x = x.transpose(-2, -1)
        y = y.transpose(-2, -1)
        sim = torch.mean(cos(x , y))
    return 1 - sim

def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x

def compute_loss(x, y, apply_mean=True, apply_log=False, type=L1_LOSS):
    x_mean = compute_mean_f0(x)
    y_mean = compute_mean_f0(y)
    if type == L1_LOSS:
        loss = F.l1_loss(x / x_mean, y / y_mean)

    elif type == PEARSON:
        if apply_log:
            x = torch.nan_to_num(torch.log10(x), nan=0, neginf=0, posinf=0)
            y = torch.nan_to_num(torch.log10(y), nan=0, neginf=0, posinf=0)
        p1 = ((x - x_mean) * (y - y_mean)).sum(dim=list(range(x.ndim)[1:]))
        p2 = torch.sqrt(torch.square(x - x_mean).sum(dim=list(range(x.ndim)[1:])) * torch.square(y - y_mean).sum(
            dim=list(range(y.ndim)[1:])))
        loss = torch.mean(1 - (p1 / p2))

    elif type == COS:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        start_dim = 1
        x = torch.flatten((x / x_mean) if apply_mean else x, start_dim=start_dim, end_dim=- 1)  # B x y
        y = torch.flatten((y / y_mean) if apply_mean else y, start_dim=start_dim, end_dim=- 1)  # B x y
        loss = 1 - torch.mean(cos(x, y))

    elif type == DOT:
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        loss = 1 - torch.mean(x * y)


    return loss

def f0_mssim_loss(x_f0, y_f0):
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    norm_x_f0 = (x_f0/x_mean)
    norm_y_f0 = (y_f0/y_mean)
    ssim_loss = torch.zeros(1, device=x_f0.device)
    batch_size = x_f0.shape[0]
    for i in [5, 10, 20]:
        window = _gaussian(i).to(x_f0.dtype).to(x_f0.device)
        #window = window.repeat(batch_size, 1)
        ssim = ssim_1D(norm_x_f0.unsqueeze(1), norm_y_f0.unsqueeze(1), window.unsqueeze(1))
        ssim_loss += (torch.ones(1, device=x_f0.device) - ssim)
    ssim_loss = ssim_loss / 3
    return ssim_loss

def _gaussian(kernel_size, sigma= 1.5 ):
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)

def ssim_1D(seq1, seq2, window, k1= 0.01, k2= 0.03):
    mu1 = F.conv1d(seq1, window, padding="same")
    mu2 = F.conv1d(seq2, window, padding="same")
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv1d(seq1 * seq1, window, padding="same") - mu1_sq
    sigma2_sq = F.conv1d(seq2 * seq2, window, padding="same") - mu2_sq
    sigma12 = F.conv1d(seq1 * seq2, window, padding="same") - mu12

    numerator1 = 2 * mu12 + k1
    numerator2 = 2 * sigma12 + k2
    denominator1 = mu1_sq + mu2_sq + k1
    denominator2 = sigma1_sq + sigma2_sq + k2
    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    return ssim_score.mean()
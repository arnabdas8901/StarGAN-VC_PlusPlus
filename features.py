import math
import torch
import librosa
import numpy as np
import torchaudio.functional as F
import torchaudio

def get_loudness(batched_waveform : torch.Tensor, n_fft: int = 124, hop_length: int = 64, sampling_rate :int = 16000 ) -> torch.Tensor:
    """
        This function calculates loudness from batched waveform. It converts the waveform to a A weighted spectrogram
        and then calculates the loudness based on librosa.feature.rms function

        batched_waveform : batched tensor. Must be a 2D tenssor, having shape b*T, where,
                           b is batch size and T is time domain sample length
    """
    window_length = n_fft
    spectrogram = torch.abs(torch.stft(batched_waveform, n_fft=n_fft, hop_length=hop_length,
                                       return_complex=True, window=torch.hann_window(window_length=window_length).to(batched_waveform.device),
                                       pad_mode='constant'))
    spectogram_db = F.amplitude_to_DB(spectrogram, multiplier=20, db_multiplier=1.0, amin=1e-5, top_db=80.0)

    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weights = librosa.A_weighting(freqs)
    a_weights = torch.from_numpy(np.expand_dims(a_weights, axis=(0, -1))).to(batched_waveform.device)

    spectogram_dba = spectogram_db + a_weights
    spectrogram_mag_a = F.DB_to_amplitude(spectogram_dba, power=0.5 , ref=1)

    spectrogram_mag_a = torch.square(spectrogram_mag_a)
    spectrogram_mag_a[:, 0, :] *= 0.5
    spectrogram_mag_a[:, -1, :] *= 0.5
    # loudness = torch.sqrt(torch.mean(torch.square(spectrogram_mag_a), dim=1))
    loudness = 2 * torch.sum(spectrogram_mag_a, dim=1) / (n_fft ** 2)
    loudness = torch.sqrt(loudness)

    return loudness

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = abs(signal_length - frames_overlap) % abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0 and rest_samples != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = torch.nn.functional.pad(signal, pad_axis, "constant", pad_value)
    frames=signal.unfold(axis, frame_length, frame_step)
    return frames

def torch_like_frame(signal, frame_length, frame_step, pad_mode="reflect" ):
    signal_dim = signal.dim()
    extended_shape = [1] * (3 - signal_dim) + list(signal.size())
    pad = int(frame_length // 2)
    input = torch.nn.functional.pad(signal.view(extended_shape), [pad, pad-1], pad_mode)
    input = input.view(input.shape[-signal_dim:])
    return input.unfold(-1, frame_length, frame_step)

def get_TEO(signal):
    trimmed_signal = signal[..., 1:-1]
    trimmed_signal_power = trimmed_signal **2

    signal_left_shift = signal[...,:-2]
    signal_right_shift = signal[...,2:]

    return  trimmed_signal_power - signal_left_shift*signal_right_shift

def _get_normed_corr(wave, sr, centre_freq, bandwidth, frame_length):
    teod = get_TEO(torchaudio.functional.bandpass_biquad(wave, sr, centre_freq, centre_freq/bandwidth))
    segmented = torch_like_frame(teod, frame_length, int(frame_length / 2)).unsqueeze(-1)
    var = torch.var(segmented, -2, keepdim=True)
    corr_mat = (segmented * torch.transpose(segmented, -2, -1) / 600) / var
    corrs = []
    for i in range(frame_length):
        corrs.append(torch.sum(torch.diagonal(corr_mat, i, dim1=-2, dim2=-1), dim=-1))

    norm_corr = torch.stack(corrs, -1)
    del teod, segmented, var, corr_mat, corrs
    return norm_corr

def get_teo_cb_Auto(waveform_real, waveform_fake, sr):
    loss = 0.0
    frame_length = int(sr*25/1000)
    cb_definition = {}
    freq_centres = [150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400]
    band_widths = [100, 100, 110, 110, 120, 140, 150, 160, 190, 210, 240, 280, 320, 380, 450, 550]
    for freq, bw in zip(freq_centres, band_widths):
        cb_definition[freq] = bw
    for freqs in cb_definition.keys():
        loss += torch.nn.functional.l1_loss(_get_normed_corr(waveform_real, sr,freqs, cb_definition[freqs], frame_length),
                                            _get_normed_corr(waveform_fake, sr, freqs, cb_definition[freqs], frame_length))

    return loss
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import sys
import yaml
import torch
import librosa
import torchaudio
import soundfile as sf
sys.path.append("/StarGAN_v2/")
from Utils.EMORECOG.train import get_data_path_list
from Utils.EMORECOG.model_alternate import build_model
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet


config_path='/StarGAN_v2/Utils/EMORECOG/config.yml'
config = yaml.safe_load(open(config_path))

batch_size = 32
device="cuda"
domain="emotions"
train_path = "/dataset/ESD/Emotional Speech Dataset (ESD)/English_sub_set/train_list.txt"
val_path = "/StarGAN_v2/Data/selected_esd_val.txt"
_, val_list = get_data_path_list(train_path, val_path)
total_samples = 0
correct_pred = 0.
# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
with open(ASR_config) as f:
    ASR_config = yaml.safe_load(f)
ASR_model_config = ASR_config['model_params']
ASR_model = ASRCNN(**ASR_model_config)
params = torch.load(ASR_path, map_location='cpu')['model']
ASR_model.load_state_dict(params)
_ = ASR_model.eval()

# load pretrained F0 model
F0_path = config.get('F0_path', False)
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load(F0_path, map_location='cpu')['net']
F0_model.load_state_dict(params)

complete_model_path = "/experiments/stargan-v2/esdall_emotion_classifier_experiment_alternate_epochs200/epoch_00196.pth"
_, Emo_classifier = build_model(F0_model=F0_model, ASR_model=ASR_model)
params = torch.load(complete_model_path, map_location='cpu')
params = params['model_ema']
_ = [Emo_classifier[key].load_state_dict(params[key]) for key in Emo_classifier]
_ = [Emo_classifier[key].eval().to(device) for key in Emo_classifier]

MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
mean, std = -4, 4
source_path = "/StarGAN_v2/output/validation/CremaD_Angry/"
methods = { "vctk20_ESD1_kur_demo_chunk2_highF0_epochs200":"Combo"}

def get_index(item):
    return int(item.split("/")[-1].split("_")[0])



for method in methods.keys():
    full_path = os.path.join(source_path, method)
    file_path = os.listdir(full_path)
    wavs_path = [os.path.join(full_path, file) for file in file_path if ".wav" in file]
    wavs_path = sorted(wavs_path, key=get_index)
    count = 0
    total = 0
    for wav_path in wavs_path:
        if ".wav" in wav_path:
            total+= 1
            index = int(wav_path.split("/")[-1].split("_")[0]) - 1
            label = int(val_list[index].split("|")[1][0])
            wave, sr = sf.read(os.path.join(full_path, wav_path))
            if sr != 24000:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            wave_tensor = torch.from_numpy(wave).float()
            mel_tensor = to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - mean) / std
            mel_length = mel_tensor.size(1)
            mels = torch.zeros((1, 80, mel_length)).float()
            mels[0, :, :mel_length] = mel_tensor
            mel = mels.unsqueeze(1).to(device)
            op, _, _ = Emo_classifier.classifier(None, None, None, mel)
            pred = torch.argmax(op, dim=-1).item()
            val = torch.max(torch.softmax(op, dim=-1), dim=-1)[0].item()
            print(wav_path, pred, val)
            if label == pred:
                count += 1
    correct = (count/total)*100
    print(method, total, count, correct)


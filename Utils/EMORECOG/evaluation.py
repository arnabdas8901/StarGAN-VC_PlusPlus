import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import yaml
import torch
sys.path.append("/project/ardas/StarGAN_v2/")
from Utils.EMORECOG.train import get_data_path_list
from Utils.EMORECOG.dataset import build_dataloader
#from Utils.EMORECOG.model import build_model
from Utils.EMORECOG.model_alternate import build_model
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet


config_path='/project/ardas/StarGAN_v2/Utils/EMORECOG/config.yml'
config = yaml.safe_load(open(config_path))

batch_size = 32
device="cuda"
domain="emotions"
train_path = "/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/English_sub_set/train_list.txt"
val_path = "/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/English_sub_set/val_list.txt"
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

#complete_model_path = "/project/ardas/experiments/stargan-v2/esdall_emotion_classifier_experiment_epochs200/epoch_00126.pth"
complete_model_path = "/project/ardas/experiments/stargan-v2/esdall_emotion_classifier_experiment_alternate_epochs200/epoch_00196.pth"
_, Emo_classifier = build_model(F0_model=F0_model, ASR_model=ASR_model)
params = torch.load(complete_model_path, map_location='cpu')
params = params['model_ema']
_ = [Emo_classifier[key].load_state_dict(params[key]) for key in Emo_classifier]
_ = [Emo_classifier[key].eval().to(device) for key in Emo_classifier]

val_dataloader = build_dataloader(val_list,
                                  batch_size=batch_size,
                                  validation=True,
                                  num_workers=2,
                                  device=device,
                                  domain=domain,
                                  random_start=False)


for i, batch in enumerate(val_dataloader):
    ### load data
    batch = [b.to(device) for b in batch]
    mel, intensity, label = batch
    batch_size = mel.size(0)
    total_samples += batch_size
    phon = torch.zeros((batch_size, 1, 192, 80), device=device, dtype=intensity.dtype )
    f0 = torch.zeros_like(intensity, device=device, dtype=intensity.dtype)
    op, _, _ = Emo_classifier.classifier(f0, intensity, phon, mel)
    pred = torch.argmax(op, dim=-1)
    correct_pred += (pred==label).sum().item()

print("Total samples ", total_samples)
print("Correctly predicted ", correct_pred )
print("Accracy ", round((correct_pred*100)/total_samples, 2),"%")
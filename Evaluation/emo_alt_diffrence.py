import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import pandas as pd
from Utils.evaluation_metrics import pitchCorr_f
from Utils.EMORECOG.model_alternate import build_model as bm
from Evaluation.utils import get_src_wave, split_source_wave

emotion_encoder_path = "/project/ardas/experiments/stargan-v2/esdall_emotion_classifier_experiment_alternate_epochs200/epoch_00196.pth"
_, emotion_encoder = bm(F0_model=None, ASR_model=None)
model_params = torch.load(emotion_encoder_path, map_location='cpu')['model_ema']['classifier']
emotion_encoder.classifier.load_state_dict(model_params)
emotion_encoder = emotion_encoder.classifier


file_name = "/scratch/ardas/Evaluation/Objective/objective_style_con_objective.csv"
target_file_name = "/scratch/ardas/Evaluation/Objective/objective_style_con_objective_update.csv"
df = pd.read_csv(file_name)


for pos in range(df.shape[0]):
    print(pos)
    source_path = df.loc[pos, "source wav"]
    try:
        source_emo = emotion_encoder.encoder.get_shared_feature(split_source_wave(get_src_wave(source_path)))
        for l in list(emotion_encoder.fc_o.modules())[1:-1]:
            source_emo = l(source_emo)
        target_emo = emotion_encoder.encoder.get_shared_feature(split_source_wave(get_src_wave(df.loc[pos, "converted wav"])))
        for l in list(emotion_encoder.fc_o.modules())[1:-1]:
            target_emo = l(target_emo)

        emo_diff = source_emo - target_emo
    except:
        print("Error", pos)
        continue

    df.loc[pos,'emo_alt_code_diff'] = torch.mean(torch.abs(emo_diff)).detach().item()

df.to_csv(target_file_name, index=False)
print("End")
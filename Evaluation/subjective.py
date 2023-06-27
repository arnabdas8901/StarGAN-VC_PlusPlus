import pandas as pd
import numpy as np
results_path = "/scratch/ardas/Evaluation/SSW/answers-final-02.05.2023.csv"
df = pd.read_csv(results_path)
df.reset_index()
trap_failed = 0
base_match_source = 0
base_match_ref = 0
ours_match_source = 0
ours_match_ref = 0
source_quality= []
base_quality = []
ours_quality = []
quality_answers = ["Answer.quality1:selections", "Answer.quality2:selections",
                   "Answer.quality3:selections", "Answer.quality4:selections",
                   "Answer.quality5:selections", "Answer.quality6:selections"]
quality_lists  =["DynamicContent.column_QUALITY_TRG1", "DynamicContent.column_QUALITY_TRG2",
                 "DynamicContent.column_QUALITY_TRG3", "DynamicContent.column_QUALITY_TRG4",
                 "DynamicContent.column_QUALITY_TRG5", "DynamicContent.column_QUALITY_TRG6"]
emotion_preservations = ["Answer.emotion_preservation1:selections", "Answer.emotion_preservation2:selections",
                         "Answer.emotion_preservation3:selections", "Answer.emotion_preservation4:selections",
                         "Answer.emotion_preservation5:selections", "Answer.emotion_preservation6:selections"]
emotion_preservation_target = ["DynamicContent.column_EMOTION_TRG1", "DynamicContent.column_EMOTION_TRG2",
                                "DynamicContent.column_EMOTION_TRG3","DynamicContent.column_EMOTION_TRG4",
                                "DynamicContent.column_EMOTION_TRG5","DynamicContent.column_EMOTION_TRG6"]
emotion_preservation_option1 = ["DynamicContent.column_EMOTION_OPT11", "DynamicContent.column_EMOTION_OPT12",
                                "DynamicContent.column_EMOTION_OPT13","DynamicContent.column_EMOTION_OPT14",
                                "DynamicContent.column_EMOTION_OPT15","DynamicContent.column_EMOTION_OPT16"]
emotion_preservation_option2 = ["DynamicContent.column_EMOTION_OPT21", "DynamicContent.column_EMOTION_OPT22",
                                "DynamicContent.column_EMOTION_OPT23","DynamicContent.column_EMOTION_OPT24",
                                "DynamicContent.column_EMOTION_OPT25","DynamicContent.column_EMOTION_OPT26"]
for pos in range(df.shape[0]):
    if df.loc[pos, "Answer.emotion_preservation7:selections"] == df.loc[pos, "DynamicContent.trapping_answer"]:
        for indx, emotion_question in enumerate(emotion_preservations):
            if df.loc[pos, emotion_question] ==1:
                if "OURS" in df.loc[pos, emotion_preservation_target[indx]]:
                    if "SRC" in df.loc[pos, emotion_preservation_option1[indx]]:
                        ours_match_source +=1
                    else:
                        ours_match_ref +=1
                else:
                    if "SRC" in df.loc[pos, emotion_preservation_option1[indx]]:
                        base_match_source +=1
                    else:
                        base_match_ref +=1
            else:
                if "OURS" in df.loc[pos, emotion_preservation_target[indx]]:
                    if "SRC" in df.loc[pos, emotion_preservation_option2[indx]]:
                        ours_match_source +=1
                    else:
                        ours_match_ref +=1
                else:
                    if "SRC" in df.loc[pos, emotion_preservation_option2[indx]]:
                        base_match_source +=1
                    else:
                        base_match_ref +=1

        for idx, qua_list in enumerate(quality_lists):
            if "SRC" in df.loc[pos, qua_list]:
                source_quality.append(6-df.loc[pos, quality_answers[idx]])
            elif "OURS" in df.loc[pos, qua_list]:
                ours_quality.append(6-df.loc[pos, quality_answers[idx]])
            else:
                base_quality.append(6-df.loc[pos, quality_answers[idx]])


    else:
        trap_failed +=1

print("trap_failed",trap_failed)
print("base_match_source", base_match_source)
print("base_match_ref", base_match_ref)
print("ours_match_source", ours_match_source)
print("ours_match_ref", ours_match_ref)
print("Source quality mean and std" , np.array(source_quality).mean(), np.array(source_quality).std())
print("Base quality mean and std" , np.array(base_quality).mean(), np.array(base_quality).std())
print("Ours quality mean and std" , np.array(ours_quality).mean(), np.array(ours_quality).std())
# StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion

## Pre-requisites
1. Python >= 3.7
2. Python requirements: 
```bash
pip install SoundFile torchaudio munch parallel_wavegan torch pydub pyyaml click librosa
```
## Training
```bash
python train.py --config_path ./Configs/config.yml
```
Please specify the training and validation data in `config.yml` file. Change `num_domains` to the number of speakers in the dataset. The data list format needs to be `filename.wav|speaker_number`, see [train_list.txt](https://github.com/yl4579/StarGANv2-VC/blob/main/Data/train_list.txt) as an example. 

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take. However, please note that `batch_size = 5` will take around 10G GPU RAM. 

## ASR & F0 Models

The pretrained F0 and ASR models are provided under the `Utils` folder. Both the F0 and ASR models are trained with melspectrograms preprocessed using [meldataset.py](https://github.com/yl4579/StarGANv2-VC/blob/main/meldataset.py), and both models are trained on speech data only. 

The ASR model is trained on English corpus, but it appears to work when training StarGANv2 models in other languages such as Japanese. The F0 model also appears to work with singing data. For the best performance, however, training your own ASR and F0 models is encouraged for non-English and non-speech data. 

You can edit the [meldataset.py](https://github.com/yl4579/StarGANv2-VC/blob/main/meldataset.py) with your own melspectrogram preprocessing, but the provided pretrained models will no longer work. You will need to train your own ASR and F0 models with the new preprocessing. 

The code for training new ASR models is available [here](https://github.com/yl4579/AuxiliaryASR) and that for training new F0 models is available [here](https://github.com/yl4579/PitchExtractor).

## References
- [clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [tosaka-m/japanese_realtime_tts](https://github.com/tosaka-m/japanese_realtime_tts)
- [keums/melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)

# BAF-Net: Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework

This is an official PyTorch implementataion of paper "Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework", which has been accepted to [INTERSPEECH 2025](https://www.isca-archive.org/interspeech_2025/kim25s_interspeech.html#). 

## Dataset preparation
The BAF-Net is trained and evaluated with Throat-Acoustic Parining Speech (TAPS) Dataset. The dataset can be accessed at [Huggingface](https://huggingface.co/datasets/yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset).

## Noise dataset preparation
The noise dataset used for data augmentation during training is the ICASSP 2023 Deep Noise Suppression Challenge dataset, which can be downloaded from [here](https://github.com/microsoft/DNS-Challenge). This dataset is used to add realistic noise to clean speech samples for robust model training.

After downloading the noise dataset, both the noise files and RIR files should be resampled to 16 kHz.  
The code for resampling is provided in `resample.py`.

```
python resample.py \
    --noise_dir PATH_TO_NOISE_DIR \
    --rir_dir PATH_TO_RIR_DIR \
    [--noise_out OUTPUT_DIR_FOR_NOISE] \
    [--rir_out OUTPUT_DIR_FOR_RIR] \
    [--sr TARGET_SAMPLE_RATE]
```


The dataset splits are defined in the following files:
- `dataset/noise_train.txt`: Training set noise files
- `dataset/noise_dev.txt`: Development set noise files  
- `dataset/noise_test.txt`: Test set noise files

Similarly, the Room Impulse Response (RIR) dataset splits are defined in:
- `dataset/rir_train.txt`: Training set RIR files
- `dataset/rir_dev.txt`: Development set RIR files
- `dataset/rir_test.txt`: Test set RIR files

These predefined splits ensure reproducibility of the experimental results.
You need to add the directory paths containing the noise and RIR data to the `config.yaml` file.
```yaml
dset:
  noise_dir: PATH_TO_NOISE_DIR
  rir_dir: PATH_TO_RIR_DIR
```

## Requirements
`pip install -r requirements.txt`

## Training BAF-Net
Training BAF-Net consists of three stages
1. Training two branches of modules: DCCRN and SE-conformer
```
# Training SE-Conformer
train.py +model=seconformer_tm

# Training DCCRN
train.py +model=dccrn
```

2. Copy the checkpoint files to the directory
```
# Copy the checkpoint files to the directory
cp outputs/seconformer_tm/best.th $PATH_TO_SEconformer_Checkpoint
cp outputs/dccrn/best.th $PATH_TO_DCCRN_Checkpoint
```

3. End-to-end training BAF-Net
```
# Training BAF-Net
train.py +model=bafnet \
    model.param.checkpoints_dccrn=$PATH_TO_DCCRN_Checkpoint \
    model.param.checkpoints_seconformer=$PATH_TO_SEconformer_Checkpoint
```


## Training Baselines
We provide training setup for baseline models

### Single-modal models
- [Deep Complex Convolution Recurrent Network (DCCRN)](https://arxiv.org/abs/2008.00264)
```
# Training DCCRN
train.py +model=dccrn
```
- [SE-Conformer](https://kakaoenterprise.github.io/papers/interspeech2021-se-conformer)
```
# Training SE-Conformer version of TM
train.py +model=seconformer_tm

# Training SE-Conformer version of AM
train.py +model=seconformer_am
```

### Multi-modal models
- [Fully Convolutional Network (FCN)](https://arxiv.org/abs/1911.09847)
```
# Training FCN version of early fusion
train.py +model=fcn_ef

# Training FCN version of late fusion
train.py +model=fcn_lf
```

- [Attention-based Fusion Densely Connected Convolutional Recurrent Network (Attention-DC-CRN)](https://ieeexplore.ieee.org/document/9746374/)
```
# Training Attention-DC-CRN version of early fusion
train.py +model=afdccrn_ef

# Training Attention-DC-CRN version of late fusion
train.py +model=afdccrn_lf
```

## Evaluate models
You can use the `evaluate.py` script to evaluate the performance of BAF-Net and other models. This script computes objective speech quality metrics such as PESQ, STOI, CSIG, CBAK, COVL, and optionally evaluates speech recognition performance (CER, WER) as well.

```bash
python evaluate.py \
    --model_config conf/model/MODEL_NAME.yaml \
    --chkpt_dir outputs/MODEL_NAME \
    --chkpt_file best.th \
    --noise_dir PATH_TO_NOISE_DIR \
    --noise_test dataset/noise_test.txt \
    --rir_dir PATH_TO_RIR_DIR \
    --rir_test dataset/rir_test.txt \
    --snr_step snr_step1 snr_step2 ... \
    --eval_stt  # Optional STT performance evaluation
```

The evaluation results are saved to the specified log file, and you can check the performance metrics at various SNR (Signal-to-Noise Ratio) levels.

## Inference
To generate enhanced speech using trained models, you can use the `enhance.py` script. This script runs the model on noisy test datasets and saves enhanced speech files (.wav) and spectrograms (.png).

```bash
python enhance.py \
    --chkpt_dir checkpoint/MODEL_NAME \
    --chkpt_file best.th \
    --noise_dir PATH_TO_NOISE_DIR \
    --noise_test dataset/noise_test.txt \
    --rir_dir PATH_TO_RIR_DIR \
    --rir_test dataset/rir_test.txt \
    --snr 0 \
    --output_dir samples/MODEL_NAME
```

The enhanced speech samples are saved in the `wavs_0dB` folder of the specified output directory, and the corresponding spectrograms are saved in the `mels_0dB` folder. Each sample includes acoustic microphone (AM) input, acoustic microphone output, throat microphone (TM) input, and the model's prediction results.

## How to cite
```
@inproceedings{kim25s_interspeech,
  title     = {{Modality-Specific Speech Enhancement and Noise-Adaptive Fusion for Acoustic and Body-Conduction Microphone Framework}},
  author    = {Yunsik Kim and Yoonyoung Chung},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {3833--3837},
  doi       = {10.21437/Interspeech.2025-2581},
  issn      = {2958-1796},
}
```
## License
BAF-Net is released under the MIT license as found in the LICENSE file.
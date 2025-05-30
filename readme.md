# 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐉 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://huggingface.co/datasets"><img alt="HuggingFace Datasets" src="https://img.shields.io/badge/datasets 2.19-yellow?style=for-the-badge&logo=huggingface&logoColor=white"></a>

This is an official PyTorch implementataion of paper "", which has been submitted to INTERSPEECH 2025. 

## Dataset download
The BAF-Net is trained and evaluated with Throat-Acoustic Parining Speech (TAPS) Dataset. The dataset can be accessed at Huggingface.


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
train.py +model=fcnef

# Training FCN version of late fusion
train.py +model=fcnlf
```

- [Attention-based Fusion Densely Connected Convolutional Recurrent Network (AFDC-CRN)](https://ieeexplore.ieee.org/document/9746374/)
```
# Training AFDC-CRN version of early fusion
train.py +model=afdccrnef

# Training AFDC-CRN version of late fusion
train.py +model=afdccrnlf
```

## Evaluate models
You can evaluate the model performance 

## How to cite

## License

BAF-Net is released under the MIT license as found in the LICENSE file.
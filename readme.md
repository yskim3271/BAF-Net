# 

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/-Pytorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/-🐉 hydra 1.3-89b8cd?style=for-the-badge&logo=hydra&logoColor=white"></a>
<a href="https://huggingface.co/datasets"><img alt="HuggingFace Datasets" src="https://img.shields.io/badge/datasets 2.19-yellow?style=for-the-badge&logo=huggingface&logoColor=white"></a>

This is an official PyTorch implementataion of paper "", which has been submitted to INTERSPEECH 2025. 

## Dataset download
The BAF-Net is trained and evaluated with Throat-Acoustic Parining Speech (TAPS) Dataset. The dataset can be accessed at Huggingface.

## Comparison with other models



## Requirements
`pip install -r requirements.txt`

## Training BAF-Net
Training BAF-Net consists of three stages
1. Training two branches of modules: DCCRN and SE-conformer
2. Load the trained weight on BAF-Net
3. End-to-end training BAF-Net

```
# First training seconformer and DCCRN
train.py --config-name=config_seconformer_tm
train.py --config-name=config_dccrn

# 
train.py --config-name=config_bafnet \
    model.param.checkpoints_dccrn=$PATH_TO_DCCRN_Checkpoint \
    model.param.checkpoints_seconformer=$PATH_TO_SEconformer_Checkpoint

```


## Training Baselines
we provide training setup for three different baselines model: DCCRN, SE-Conformer, VibVoice. The DCCRN and SE-conformer is 


## Evaluate models
You can evaluate the model performance with setting flag `eval_only=True`, and insert checkpoint directory path `continue_from=$checkpoint_dir_path$`

```
python train.py --config-name=$model_config eval_only=True continue_from=$checkpoint_dir_path$
```

## How to cite

## License

BAF-Net is released under the MIT license as found in the LICENSE file.
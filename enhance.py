

import os
import torch
import torchaudio

from matplotlib import pyplot as plt
from utils import LogProgress, mel_spectrogram

def save_wavs(wavs_dict, filepath, sr=16_000):
    for i, (key, wav) in enumerate(wavs_dict.items()):
        torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        
def save_mels(wavs_dict, filepath):
    num_mels = len(wavs_dict)
    figure, axes = plt.subplots(num_mels, 1, figsize=(10, 10))
    figure.set_tight_layout(True)
    figure.suptitle(filepath)
    
    for i, (key, wav) in enumerate(wavs_dict.items()):
        mel = mel_spectrogram(wav, device='cpu', sampling_rate=16_000)
        axes[i].imshow(mel.squeeze().numpy(), aspect='auto', origin='lower')
        axes[i].set_title(key)
    
    figure.savefig(filepath)
    plt.close(figure)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)

def enhance_multiple_snr(args, model, dataloader_list, epoch=None, local_out_dir=None):
    for snr, data_loader in dataloader_list.items():
        enhance(args, model, data_loader, logger, snr, epoch, local_out_dir)

def enhance(args, model, data_loader, logger, snr, epoch=None, local_out_dir=None):
    model.eval()
    
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
        
    suffix = f"_epoch{epoch+1}" if epoch is not None else ""
    
    iterator = LogProgress(logger, data_loader, name=f"Enhance on {snr}dB")
    outdir_mels= os.path.join(out_dir, f"mels" + suffix + f"_{snr}dB")
    outdir_wavs= os.path.join(out_dir, f"wavs" + suffix + f"_{snr}dB")
    os.makedirs(outdir_mels, exist_ok=True)
    os.makedirs(outdir_wavs, exist_ok=True)
    
    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Generate enhanced files")
        for data in iterator:
            # Get batch data (batch, channel, time)
            tm, noisy_am, clean_am, id, text = data
                        
            if args.model.input_type == "am":
                clean_am_hat = model(noisy_am.to(args.device))
            elif args.model.input_type == "tm":
                clean_am_hat = model(tm.to(args.device))
            elif args.model.input_type == "am+tm":
                clean_am_hat = model(tm.to(args.device), noisy_am.to(args.device))
            else:
                raise ValueError("Invalid model input type argument")
            
            tm = tm.squeeze(1).cpu()
            clean_am = clean_am.squeeze(1).cpu()
            noisy_am = noisy_am.squeeze(1).cpu()
            clean_am_hat = clean_am_hat.squeeze(1).cpu()
                        
            wavs_dict = {
                "tm": tm,
                "noisy_am": noisy_am,
                "clean_am": clean_am,
                "clean_am_hat": clean_am_hat,
            }
            
            save_wavs(wavs_dict, os.path.join(outdir_wavs, id[0]))
            save_mels(wavs_dict, os.path.join(outdir_mels, id[0]))


if __name__=="__main__":
    import logging
    import logging.config
    import argparse
    import importlib
    from data import TAPSnoisytdataset, StepSampler
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name. default is best.th")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory.")
    parser.add_argument("--noise_test", type=str, required=True, help="List of noise files for testing.")
    parser.add_argument("--rir_dir", type=str, required=True, help="Path to the RIR directory.")
    parser.add_argument("--rir_test", type=str, required=True, help="List of RIR files for testing.")
    parser.add_argument("--snr", type=float, default=0, help="Signal to noise ratio. default is 0 dB")
    parser.add_argument("--reverb_prop", type=float, default=0, help="Reverberation proportion. default is 0")
    parser.add_argument("--target_dB_FS", type=float, default=-26, help="Target dB FS. default is -26")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory for enhanced samples. default is samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device (cuda or cpu).")
    
    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    chkpt_file = args.chkpt_file
    device = args.device
    local_out_dir = args.output_dir

    conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "config.yaml"))
    hydra_conf = OmegaConf.load(os.path.join(chkpt_dir, '.hydra', "hydra.yaml"))
    del hydra_conf.hydra.job_logging.handlers.file
    hydra_conf.hydra.job_logging.root.handlers = ['console']
    logging_conf = OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)
    
    
    logging.config.dictConfig(logging_conf)
    logger = logging.getLogger(__name__)
    conf.device = device
    
    model_args = conf.model
    model_name = model_args.model_name
    module = importlib.import_module("models."+ model_name)
    model_class = getattr(module, model_name)
    
    model = model_class(**model_args.param).to(device)
    chkpt = torch.load(os.path.join(chkpt_dir, chkpt_file), map_location=device)
    model.load_state_dict(chkpt['model'])
    
    testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    
    tt_dataset = TAPSnoisytdataset(datapair_list=testset,
                                   noise_list=args.noise_test,
                                   rir_list=args.rir_test,
                                   snr_range=[args.snr, args.snr],
                                   reverb_proportion=args.reverb_prop,
                                   target_dB_FS=args.target_dB_FS,
                                   with_id=True,
                                   with_text=True)

    
    tt_loader = DataLoader(
        dataset=tt_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=StepSampler(len(tt_dataset), step=100)
    )
    
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {local_out_dir}")
    os.makedirs(local_out_dir, exist_ok=True)
    
    enhance(model=model,
            data_loader=tt_loader,
            args=conf,
            epoch=None,
            logger=logger,
            local_out_dir=local_out_dir
            )
        
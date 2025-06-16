# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik
import torch
import nlptutti as sarmetric
import numpy as np
from pesq import pesq
from pystoi import stoi
from metric_helper import wss, llr, SSNR, trim_mos
from utils import bold, LogProgress
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_stts(args, logger, enhanced):

    cer, wer = 0, 0
    model_id = "ghost613/whisper-large-v3-turbo-korean"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(args.device)
    
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float32,
        device=args.device,)
    
    iterator = LogProgress(logger, enhanced, name="STT Evaluation")
    for wav, text in iterator:
        with torch.no_grad():
            transcription = pipe(wav, generate_kwargs={"num_beams": 1, "max_length": 100})['text']

        cer += sarmetric.get_cer(text, transcription, rm_punctuation=True)['cer']
        wer += sarmetric.get_wer(text, transcription, rm_punctuation=True)['wer']
    
    cer /= len(enhanced)
    wer /= len(enhanced)
    
    return cer, wer


    
## Code modified from https://github.com/wooseok-shin/MetricGAN-plus-pytorch/tree/main
def compute_metrics(target_wav, pred_wav, fs=16000):
    
    Stoi = stoi(target_wav, pred_wav, fs, extended=False)
    Pesq = pesq(ref=target_wav, deg=pred_wav, fs=fs)
        
    alpha = 0.95
    ## Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    
    ## Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])
    
    ## Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, 16000)
    segSNR = np.mean(segsnr_mean)
    
    ## Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * Pesq - 0.009 * wss_dist
    Csig = float(trim_mos(Csig))
    
    ## Cbak
    Cbak = 1.634 + 0.478 * Pesq - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)

    ## Covl
    Covl = 1.594 + 0.805 * Pesq - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    
    return Pesq, Stoi, Csig, Cbak, Covl



def evaluate(args, model, data_loader_list, logger, epoch=None):
    

    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""

    metrics = {}
    model.eval()
    for snr, data_loader in data_loader_list.items():
        iterator = LogProgress(logger, data_loader, name=f"Evaluate on {snr}dB")
        enhanced = []
        results  = []
        with torch.no_grad():
            for data in iterator:
                tm, noisy_am, clean_am, id, text = data
                
                if args.model.input_type == "am":
                    clean_am_hat = model(noisy_am.to(args.device))
                elif args.model.input_type == "tm":
                    clean_am_hat = model(tm.to(args.device))
                elif args.model.input_type == "am+tm":
                    clean_am_hat = model(tm.to(args.device), noisy_am.to(args.device))
                else:
                    raise ValueError("Invalid model input type argument")

                clean_am_hat = clean_am_hat.squeeze().cpu().numpy()
                clean_am = clean_am.squeeze().cpu().numpy()
                
                if clean_am_hat.shape[0] > clean_am.shape[0]:
                    leftover = clean_am_hat.shape[0] - clean_am.shape[0]
                    clean_am_hat = clean_am_hat[leftover//2:-leftover//2]
                elif clean_am_hat.shape[0] < clean_am.shape[0]:
                    raise ValueError("Enhanced signal is shorter than clean signal")

                enhanced.append((clean_am_hat, text[0]))
                results.append(compute_metrics(clean_am, clean_am_hat))
        
        results = np.array(results)
        pesq, stoi, csig, cbak, covl = np.mean(results, axis=0)
        metrics[f'{snr}dB'] = {
            "pesq": pesq,
            "stoi": stoi,
            "csig": csig,
            "cbak": cbak,
            "covl": covl
        }
        logger.info(bold(f"{prefix}Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))
                
        if args.eval_stt:
            cer, wer = get_stts(args, logger, enhanced)
            metrics[f'{snr}dB']['cer'] = cer
            metrics[f'{snr}dB']['wer'] = wer
            logger.info(bold(f"{prefix}Performance on {snr}dB: CER={cer:.4f}, WER={wer:.4f}"))
   
    return metrics

if __name__=="__main__":
    import os
    import logging
    import argparse
    import importlib
    from data import TAPSnoisytdataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from datasets import load_dataset, concatenate_datasets
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--chkpt_dir", type=str, default='.', help="Path to the checkpoint directory. default is current directory")
    parser.add_argument("--chkpt_file", type=str, default="best.th", help="Checkpoint file name. default is best.th")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the noise directory.")
    parser.add_argument("--noise_test", type=str, required=True, help="List of noise files for testing.")
    parser.add_argument("--rir_dir", type=str, required=True, help="Path to the RIR directory.")
    parser.add_argument("--rir_test", type=str, required=True, help="List of RIR files for testing.")
    parser.add_argument("--test_augment_numb", type=int, default=2, help="Number of test augmentations. default is 2")
    parser.add_argument("--snr_step", nargs="+", type=int, required=True, help="Signal to noise ratio. default is 0 dB")
    parser.add_argument("--reverb_proportion", type=float, default=0.5, help="Reverberation proportion. default is 0.5")
    parser.add_argument("--target_dB_FS", type=float, default=-25, help="Target dB FS. default is -25")
    parser.add_argument("--target_dB_FS_floating_value", type=float, default=0, help="Target dB FS floating value. default is 0")
    parser.add_argument("--silence_length", type=float, default=0.2, help="Silence length. default is 0.2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Specifies the device (cuda or cpu).")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers. default is 5")
    parser.add_argument("--log_file", type=str, default="output.log", help="Log file name. default is output.log")
    parser.add_argument("--eval_stt", default=False, action="store_true", help="Evaluate STT performance")
    
    args = parser.parse_args()
    chkpt_dir = args.chkpt_dir
    chkpt_file = args.chkpt_file
    device = args.device

    log_file = args.log_file
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    conf = OmegaConf.load(args.model_config)
    conf.model = OmegaConf.load(args.model_config)
    conf.device = device
    conf.eval_stt = args.eval_stt
    
    model_args = os.path.basename(args.model_config)
    model_lib = conf.model_lib
    model_class = conf.model_class
    module = importlib.import_module("models."+ model_lib)
    model_class = getattr(module, model_class)
    
    model = model_class(**conf.param).to(device)
    chkpt = torch.load(os.path.join(chkpt_dir, chkpt_file), map_location=device)
    model.load_state_dict(chkpt['model'])
    tm_only = conf.input_type == "tm"
    
    testset = load_dataset("yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset", split="test")
    testset_list = [testset] * args.test_augment_numb
    testset = concatenate_datasets(testset_list)
    
    ev_loader_list = {}

    noise_test_list = [os.path.join(args.noise_dir, line.strip()) for line in open(args.noise_test, "r")]
    rir_test_list = [os.path.join(args.rir_dir, line.strip()) for line in open(args.rir_test, "r")]

    for fixed_snr in args.snr_step:
        ev_dataset = TAPSnoisytdataset(
            datapair_list= testset,
            noise_list= noise_test_list,
            rir_list= rir_test_list,
            snr_range= [fixed_snr, fixed_snr],
            reverb_proportion=args.reverb_proportion,
            target_dB_FS=args.target_dB_FS,
            target_dB_FS_floating_value=args.target_dB_FS_floating_value,
            silence_length=args.silence_length,
            deterministic=True,
            sampling_rate=16000,
            with_id=True,
            with_text=True,
            tm_only=tm_only,
        )

        ev_loader = DataLoader(
            dataset=ev_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True
        )

        ev_loader_list[f"{fixed_snr}"] = ev_loader

    logger.info(f"Model: {model_class.__name__}")
    logger.info(f"Input type: {conf.input_type}")
    logger.info(f"Checkpoint: {chkpt_dir}")
    logger.info(f"Device: {device}")
    
    evaluate(args=conf,
            model=model,
            data_loader_list=ev_loader_list,
            logger=logger,
            epoch=None)

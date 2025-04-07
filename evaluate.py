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



def evaluate(args, model, data_loader_list, epoch, logger):
        
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
        logger.info(bold(f"Epoch {epoch+1}, Performance on {snr}dB: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}"))
                
        if args.eval_stt:
            cer, wer = get_stts(args, logger, enhanced)
            metrics[f'{snr}dB']['cer'] = cer
            metrics[f'{snr}dB']['wer'] = wer
            logger.info(bold(f"Epoch {epoch+1}, Performance on {snr}dB: CER={cer:.4f}, WER={wer:.4f}"))
   
    return metrics



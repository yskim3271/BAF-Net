import torch
import torch.nn as nn
from models.stft import ConvSTFT, ConviSTFT
from models.seconformer import seconformer
from models.dccrn import dccrn

def normal_energy(spec: torch.Tensor, eps: float = 1e-8):
    """
    Normalize the energy of the spectrogram.
    
    Args:
        spec (torch.Tensor): Input spectrogram tensor.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        torch.Tensor: Normalized spectrogram.
        torch.Tensor: Energy of the input spectrogram.
    """
    real = spec[:, 0, :, :]  # [B, F, T]
    imag = spec[:, 1, :, :]  # [B, F, T]
    
    power = real**2 + imag**2            # [B, F, T]
    energy = power.mean(dim=[1,2], keepdim=True).sqrt()  
    spec_norm = spec / (energy + eps)

    return spec_norm, energy

class bafnet(torch.nn.Module):
    def __init__(self, 
                 depth, 
                 channels, 
                 kernel_size, 
                 args_seconformer, 
                 args_dccrn, 
                 checkpoint_seconformer=None,
                 checkpoint_dccrn=None,
                 ):
        super(bafnet, self).__init__()
        
        self.depth = depth
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
                
        self.seconformer = seconformer(
            **args_seconformer
        )
        self.dccrn = dccrn(
            **args_dccrn
        )
        
        if checkpoint_dccrn is not None:
            self.dccrn.load_state_dict(torch.load(checkpoint_dccrn)['model'])
        
        if checkpoint_seconformer is not None:
            self.seconformer.load_state_dict(torch.load(checkpoint_seconformer)['model'])            
        
        self.stft = ConvSTFT(
            args_dccrn.win_len,
            args_dccrn.win_inc,
            args_dccrn.fft_len,
            args_dccrn.win_type,
            feature_type='complex',
            fix=True
        )
        self.istft = ConviSTFT(
            args_dccrn.win_len,
            args_dccrn.win_inc,
            args_dccrn.fft_len,
            args_dccrn.win_type,
            feature_type='complex',
            fix=True
        )
        
        estimator = []
        channels = 1
        for i in range(self.depth - 1):
            estimator += [
                nn.Conv2d(in_channels=channels, 
                          out_channels=self.channels, 
                          kernel_size=self.kernel_size, 
                          stride=1, 
                          padding=self.padding),
                nn.BatchNorm2d(self.channels),
                nn.PReLU(),
            ]
            channels = self.channels
        estimator += [
            nn.Conv2d(in_channels=channels, 
                      out_channels=1, 
                      kernel_size=self.kernel_size, 
                      stride=1, 
                      padding=self.padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        ]
        
        self.ratio_estimator = nn.Sequential(*estimator)
        
        estimator = []
        
        
        self.init_estimator()
        
    def init_estimator(self):
        for m in self.ratio_estimator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.ratio_estimator.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, tm, am, lens=False):
                        
        in_len = tm.size(-1)
        
        tm_wav = self.seconformer(tm)
        
        mask_mags, am_spec, am_wav = self.dccrn(am, lens=True)

        tm_spec = self.stft(tm_wav)

        tm_spec, tm_energy = normal_energy(tm_spec)
        am_spec, am_energy = normal_energy(am_spec)

        mask_mags = mask_mags.unsqueeze(1)
        
        alpha = self.ratio_estimator(mask_mags)
        alpha = alpha.squeeze(1)
        
        alpha = alpha.repeat(1, 2, 1)  # [B, 2*F, T]
        
        est_spec = tm_spec * alpha + am_spec * (1 - alpha)
        
        rms_target = (tm_energy + am_energy) / 2.0
        
        est_spec = rms_target * est_spec
        
        est_wav = self.istft(est_spec)
                
        out_len = est_wav.size(-1)
        
        if out_len > in_len:
            leftover = out_len - in_len 
            est_wav = est_wav[..., leftover//2:-(leftover//2)]
            am_wav = am_wav[..., leftover//2:-(leftover//2)]
        
        if lens == True:
            return mask_mags, alpha, tm_wav, am_wav, est_wav
        
        return est_wav
        
        
        
        
        
        

        
        
        
        
        
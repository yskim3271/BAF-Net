import torch
import torch.nn as nn
import torch.nn.functional as F
from models.afdccrn_module import DC_CRN, AttentionFusion
from models.stft import ConvSTFT, ConviSTFT

class afdccrn_ef(nn.Module):
    """
    Early Fusion Model based on Fig 2b.
    Attention -> DC-CRN
    """
    def __init__(self,
                 window_size=512,
                 hop_size=256,
                 fft_length=512,
                 win_type='hann',
                 channels=[16, 32, 64, 128, 256],
                 dcblock_depth=4,
                 dcblock_growth_rate=8,
                 stride=(2, 1),
                 kernel_size=(4, 1),
                 lstm_groups=2,
                 lstm_layers=2,
                 lstm_bidirectional=True,
                 lstm_rearrange=False,
    ):
        super(afdccrn_ef, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_length = fft_length
        self.win_type = win_type
        self.channels = channels
        self.dcblock_depth = dcblock_depth
        self.dcblock_growth_rate = dcblock_growth_rate
        self.stride = stride
        self.kernel_size = kernel_size
        self.lstm_groups = lstm_groups
        self.lstm_layers = lstm_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_rearrange = lstm_rearrange

        # Attention Fusion Module
        self.attention_fusion = AttentionFusion(2)

        # DC-CRN Module
        self.dc_crn = DC_CRN(
            input_dim=1,
            channels=self.channels,
            dcblock_depth=self.dcblock_depth,
            dcblock_growth_rate=self.dcblock_growth_rate,
            stride=self.stride,
            kernel_size=self.kernel_size,
            lstm_groups=self.lstm_groups,
            lstm_layers=self.lstm_layers,
            lstm_bidirectional=self.lstm_bidirectional,
            lstm_rearrange=self.lstm_rearrange
        )

        self.stft = ConvSTFT(
            self.window_size,
            self.hop_size,
            self.fft_length,
            self.win_type,
            'complex',
            fix=True,
        )
        self.istft = ConviSTFT(
            self.window_size,
            self.hop_size,
            self.fft_length,
            self.win_type,
            'complex',
            fix=True,
        )
    
    def forward(self, x1, x2):
        
        in_len = x1.size(-1)
        spec1 = self.stft(x1)
        spec2 = self.stft(x2)
        
        spec1_real = spec1[:, :self.fft_length // 2 + 1, :]
        spec1_imag = spec1[:, self.fft_length // 2 + 1:, :]
        
        spec2_real = spec2[:, :self.fft_length // 2 + 1, :]
        spec2_imag = spec2[:, self.fft_length // 2 + 1:, :]
        
        spec1 = torch.stack((spec1_real, spec1_imag), dim=1)
        spec2 = torch.stack((spec2_real, spec2_imag), dim=1)
        
        spec1 = spec1[:, :, 1:, :]
        spec2 = spec2[:, :, 1:, :]
        
        fused = self.attention_fusion(spec1, spec2)
        out = self.dc_crn(fused)
        
        out_real = out[:, 0, :, :]
        out_imag = out[:, 1, :, :]
        
        out = torch.cat((out_real, out_imag), dim=1)
        
        out = self.istft(out)
        
        out_len = out.size[-1]
        
        if out_len > in_len:
            leftover = out_len - in_len 
            out = out[..., leftover//2:-(leftover//2)]
        
        return out
    
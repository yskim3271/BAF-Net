import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


def ComplextoMagPhase(real, imag):
    mags = torch.sqrt(real ** 2 + imag ** 2)
    phase = torch.atan2(imag, real)
    return mags, phase

def MagPhasetoComplex(mags, phase):
    real = mags * torch.cos(phase)
    imag = mags * torch.sin(phase)
    return torch.cat([real, imag], 1)

def SpectoComplex(spec):
    real = spec[:, :spec.shape[1]//2, :]
    imag = spec[:, spec.shape[1]//2:, :]
    return real, imag

def SpecToMagPhase(spec):
    real, imag = SpectoComplex(spec)
    return ComplextoMagPhase(real, imag)

class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        
        pad_len = inputs.size(-1) % self.stride
        if pad_len != 0:
            inputs = F.pad(inputs, [(self.stride - pad_len) // 2, (self.stride - pad_len) // 2])
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)  # x = torch.chunk(x, n, dim)  >> x의 dim 차원을 n개씩 잘라서 뽑아옴
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)   # torch.cat : 차원 늘리기
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs
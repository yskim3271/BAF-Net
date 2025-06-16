"""
dccrn: Deep complex convolution recurrent network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stft import ConvSTFT, ConviSTFT, complex_cat



def normal_energy(spec: torch.Tensor, eps: float = 1e-8):
    
    real = spec[:, :spec.shape[1]//2, :]  # [B, F, T]
    imag = spec[:, spec.shape[1]//2:, :]  # [B, F, T]
    power = real**2 + imag**2            # [B, F, T]
    energy = power.mean(dim=[1,2], keepdim=True).sqrt()  
    spec_norm = spec / (energy + eps)

    return spec_norm, energy


class ComplexConv2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=1,
            groups=1,
            causal=True,
            complex_axis=1,
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                   padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            causal=False,
            complex_axis=1,
            groups=1
    ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size, self.stride,
                                            padding=self.padding, output_padding=output_padding, groups=self.groups)
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):

        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real, )
            imag2imag = self.imag_conv(imag, )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out

class NavieComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size//2
        self.rnn_units = hidden_size//2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        if bidirectional:
            bidirectional=2
        else:
            bidirectional=1
        if projection_dim is not None:
            self.projection_dim = projection_dim//2 
            self.r_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs,list):
            real, imag = inputs 
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs,-1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        #print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]
    
    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

class DCCRN(nn.Module):

    def __init__(
            self,
            rnn_layers=2,
            rnn_units=256,
            win_len=400,
            win_inc=100,
            fft_len=512,
            win_type='hann',
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256],
            use_clstm=True
    ):
        super(DCCRN, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        self.kernel_num = [2] + kernel_num
        self.use_clstm = use_clstm

        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm: 
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                        NavieComplexLSTM(
                        input_size= hidden_dim*self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim= hidden_dim*self.kernel_num[-1] if idx == rnn_layers-1 else None,
                        )
                    )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                    input_size= hidden_dim*self.kernel_num[-1],
                    hidden_size=self.rnn_units,
                    num_layers=2,
                    dropout=0.0,
                    bidirectional=bidirectional,
                    batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim*self.kernel_num[-1])

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                    )
                )

        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, lens=False):
        
        in_len = inputs.size(-1)
        
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_mags = spec_mags


        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan2(
            imag_phase,
            real_phase
        )

        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.clamp_(out_wav, -1, 1)

        out_len = out_wav.size(-1)
        
        if out_len > in_len:
            leftover = out_len - in_len 
            out_wav = out_wav[..., leftover//2:-(leftover//2)]
                
        if lens == True:
            return mask_mags, out_spec, out_wav
        else:
            return out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params
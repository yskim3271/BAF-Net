"""
dccrn: Deep complex convolution recurrent network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module_dccrn import ConvSTFT, ConviSTFT, \
    ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm


class dccrn(nn.Module):

    def __init__(
            self,
            rnn_layers=2,
            rnn_units=256,
            win_len=400,
            win_inc=100,
            fft_len=512,
            win_type='hann',
            use_clstm=True,
            use_cbn=False,
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256]
    ):
        '''

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(dccrn, self).__init__()

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
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

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
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
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
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

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

        # out_wav = torch.squeeze(out_wav, 1)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stft import ConvSTFT, ConviSTFT

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(GatedConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding
                               )
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding
                               )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GatedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0,0)):
        super(GatedConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


# Dense Connected Block
class DCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, growth_rate=8, stride=(2,1), kernel_size=(4,1), encode=False):
        super(DCBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.stride = stride
        self.kernel_size = kernel_size
        
        
        channels = in_channels
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, growth_rate, kernel_size, padding=(kernel_size[0]//2, 0)),
                nn.BatchNorm2d(growth_rate),
                nn.PReLU()
            ))
            channels += growth_rate
        if encode:
            self.gated_conv = GatedConv2d(channels, out_channels, kernel_size, stride=stride, padding=(stride[0]//2, 0))
        else:
            self.gated_conv = GatedConvTranspose2d(channels, out_channels, kernel_size, stride=stride, padding=(stride[0]//2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        skip = x
        out = x
        for layer in self.layers:
            out = layer(out)
            out = out[:,:,:skip.shape[-2], :]
            out = torch.cat([skip, out], dim=1)
            skip = out
        
        out = self.gated_conv(out)
        out = self.bn(out)
        out = self.prelu(out)
        return out

# Attention Fusion Module
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
    
    def forward(self, input1, input2):
        
        input_sum = input1 + input2
        br1_out = self.branch1(input_sum)
        pooled = torch.mean(input_sum, dim=(2, 3), keepdim=True)
        pooled_expanded = pooled.expand_as(br1_out)
        br2_out = self.branch2(pooled_expanded)
        attn = torch.sigmoid(br1_out + br2_out)
        out = input1 * attn + input2 * (1 - attn)
        
        return out


class GLSTM(nn.Module):
    def __init__(
        self, hidden_size=1024, groups=2, layers=2, bidirectional=False, rearrange=False
    ):
        """Grouped LSTM.

        Based on the paper: Efficient Sequence Learning with Group Recurrent Networks; Gao et al., 2018

        Args:
            hidden_size (int): Total hidden size of all LSTMs in the grouped LSTM layer  
                → Each LSTM's hidden size = hidden_size // groups
            groups (int): Number of groups (one LSTM per group)
            layers (int): Number of grouped LSTM layers
            bidirectional (bool): Whether to use bidirectional LSTM (BLSTM)  
            rearrange (bool): Whether to apply rearrange operation after each grouped LSTM layer
        """
        super().__init__()

        # hidden_size must be divisible by groups
        assert hidden_size % groups == 0, (hidden_size, groups)
        # hidden size allocated to each group's LSTM
        hidden_size_t = hidden_size // groups
        if bidirectional:
            # For bidirectional LSTM, hidden_size_t must be divisible by 2
            assert hidden_size_t % 2 == 0, hidden_size_t

        self.groups = groups
        self.layers = layers
        self.rearrange = rearrange

        # Create ModuleList to store LSTM and LayerNorm for each layer
        self.lstm_list = nn.ModuleList()
        # Create LayerNorm for each layer, each LayerNorm is applied to the last dimension hidden_size
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        
        # Configure each grouped LSTM layer
        for layer in range(layers):
            # Create individual LSTM for each group and store in ModuleList
            self.lstm_list.append(
                nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=hidden_size_t, 
                            # If bidirectional: hidden_size_t // 2, otherwise: hidden_size_t is the hidden dimension
                            hidden_size=hidden_size_t // 2 if bidirectional else hidden_size_t,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional,
                        )
                        for _ in range(groups)
                    ]
                )
            )

    def forward(self, x):
        """Grouped LSTM forward.

        Args:
            x (torch.Tensor): Input tensor, shape: (B, C, T, D)
        Returns:
            out (torch.Tensor): Output tensor, shape: (B, C, T, D)
        """
        # 1. Input x: (B, C, T, D)
        out = x

        # 2. Swap C and T dimensions → shape: (B, T, C, D)
        out = out.transpose(1, 2).contiguous()

        # 3. Flatten (B, T, C, D) to 3D tensor  
        #    Here we merge C and D dimensions to get shape: (B, T, C*D)
        #    Note: C*D must equal hidden_size!
        B, T = out.size(0), out.size(1)
        out = out.view(B, T, -1).contiguous()  # shape: (B, T, hidden_size)

        # 4. Split last dimension into groups → each chunk shape: (B, T, hidden_size // groups)
        out = torch.chunk(out, self.groups, dim=-1)

        # 5. Process the first grouped LSTM layer  
        #    Pass each split tensor to its corresponding group's LSTM  
        #    LSTM output is (B, T, hidden_size_t) (includes both directions if bidirectional)
        #    Use list comprehension to collect each group's result and stack along a new dimension
        out = torch.stack(
            [self.lstm_list[0][i](out[i])[0] for i in range(self.groups)], dim=-1
        )  # result shape: (B, T, hidden_size_t, groups)

        # 6. Flatten last two dimensions → shape: (B, T, hidden_size_t * groups) = (B, T, hidden_size)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        # 7. Apply Layer Normalization to first layer → shape preserved: (B, T, hidden_size)
        out = self.ln[0](out)

        # 8. Process remaining grouped LSTM layers starting from the second layer
        for layer in range(1, self.layers):
            # (Optional) If rearrange flag is True, reorder the groups
            if self.rearrange:
                # Current shape: (B, T, hidden_size)
                # reshape: (B, T, groups, hidden_size // groups)
                # transpose: Swap last two dimensions → (B, T, hidden_size // groups, groups)
                # Finally flatten back to (B, T, hidden_size)
                out = (
                    out.reshape(B, T, self.groups, -1)
                    .transpose(-1, -2)
                    .contiguous()
                    .view(B, T, -1)
                )
            # 8-1. Split current output by groups  
            #     Each split tensor shape: (B, T, hidden_size // groups)
            out_chunks = torch.chunk(out, self.groups, dim=-1)

            # 8-2. Process each group with its LSTM → each output shape: (B, T, hidden_size_t)
            #      Concatenate all group outputs (dim=-1) → (B, T, hidden_size)
            out = torch.cat(
                [self.lstm_list[layer][i](out_chunks[i])[0] for i in range(self.groups)],
                dim=-1,
            )
            # 8-3. Apply Layer Normalization for this layer → shape preserved: (B, T, hidden_size)
            out = self.ln[layer](out)

        # 9. Transform final output tensor shape  
        #    Current shape: (B, T, hidden_size)
        #    Convert back to 4D tensor to match original input format  
        #    x.size(1) is the original C dimension, so D becomes hidden_size / C
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()  # shape: (B, T, C, D)
        # 10. Swap C and T dimensions again → final output shape: (B, C, T, D)
        out = out.transpose(1, 2).contiguous()

        return out


class DC_CRN(nn.Module):
    """
    Densely Connected Convolutional Recurrent Network (DC-CRN) based on Fig 2a.
    Assumptions:
    - 5 Encoder/Decoder layers.
    - Channels: [in_c, 16, 32, 64, 128, 256] -> LSTM -> [256, 128, 64, 32, 16, out_c]
    - Stride (1, 2) for frequency down/up sampling in Conv/TransposedConv.
    - LSTM: 2 layers, 128 hidden units/direction, 2 groups.
    - Skip connections: 1x1 Conv + Concatenation before decoder block.
    - Final output split channels for Real/Imaginary parts.
    """
    def __init__(self,
                 input_dim=1,
                 feature_dim=256,
                 channels=[16, 32, 64, 128, 256],
                 dcblock_depth=4,
                 dcblock_growth_rate=8,
                 kernel_size=(4, 1),
                 stride=(2, 1),
                 lstm_groups=2,
                 lstm_layers=2,
                 lstm_bidirectional=True,
                 lstm_rearrange=False,
                 ):
        super().__init__()
        
        
        self.input_dim = input_dim
        
        self.channels = [input_dim*2] + channels
        self.lstm_groups = lstm_groups
        self.lstm_layers = lstm_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_rearrange = lstm_rearrange
        

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_pathway = nn.ModuleList()
        
        for i in range(len(self.channels) - 1):
            encode = DCBlock(
                self.channels[i],
                self.channels[i + 1],
                depth=dcblock_depth,
                growth_rate=dcblock_growth_rate,
                stride=stride,
                kernel_size=kernel_size,
                encode=True
            )
            
            decode = DCBlock(
                self.channels[i + 1] * 2,
                self.channels[i],
                depth=dcblock_depth,
                growth_rate=dcblock_growth_rate,
                stride=stride,
                kernel_size=kernel_size,
                encode=False
            )
            self.encoders.append(encode)
            self.decoders.insert(0, decode)
            self.skip_pathway.append(
                nn.Conv2d(
                    self.channels[i + 1],
                    self.channels[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        
        lstm_input_dim = feature_dim // (stride[0] ** (len(self.channels) - 1)) * self.channels[-1]        
        
        self.lstm = GLSTM(
            hidden_size= lstm_input_dim,
            groups=self.lstm_groups,
            layers=self.lstm_layers,
            bidirectional=self.lstm_bidirectional,
            rearrange=self.lstm_rearrange,
        )
        
        self.fc_real = nn.Linear(in_features=feature_dim, out_features=feature_dim)
        self.fc_imag = nn.Linear(in_features=feature_dim, out_features=feature_dim)

    def forward(self, x):
        # x shape: [B, C_in, T, F]
        out = x
        skips = []
        for idx, layer in enumerate(self.encoders):
            out = layer(out)
            skip = self.skip_pathway[idx](out)
            skips.insert(0, skip)
        
        out = out.permute(0, 1, 3, 2).contiguous()
        out = self.lstm(out)        
        out = out.permute(0, 1, 3, 2).contiguous()
        
        for idx in range(len(self.decoders)):
            skip = skips[idx]
            out = torch.cat((out, skip), dim=1)
            out = self.decoders[idx](out)

        out_real = out[:, 0, :, :]
        out_imag = out[:, 1, :, :]
        
        out_real = out_real.permute(0, 2, 1).contiguous()
        out_imag = out_imag.permute(0, 2, 1).contiguous()
        
        out_real = self.fc_real(out_real)
        out_imag = self.fc_imag(out_imag)
        
        out_real = out_real.permute(0, 2, 1).contiguous()
        out_imag = out_imag.permute(0, 2, 1).contiguous()
        
        out_real = F.pad(out_real, [0, 0, 1, 0])
        out_imag = F.pad(out_imag, [0, 0, 1, 0])
        
        out = torch.stack((out_real, out_imag), dim=1)
                        
        return out
    

class Attention_DC_CRN_EarlyFusion(nn.Module):
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
        super(Attention_DC_CRN_EarlyFusion, self).__init__()
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
        self.dc_crn1 = DC_CRN(
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
        self.dc_crn2 = DC_CRN(
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
        
        out1 = self.dc_crn1(spec1)
        out2 = self.dc_crn2(spec2)
        
        out = self.attention_fusion(out1, out2)
        
        out_real = out[:, 0, :, :]
        out_imag = out[:, 1, :, :]
        
        out = torch.cat((out_real, out_imag), dim=1)
        
        out = self.istft(out)
        
        out_len = out.shape[-1]
        
        if out_len > in_len:
            leftover = out_len - in_len 
            out = out[..., leftover//2:-(leftover//2)]
        
        return out


class Attention_DC_CRN_LateFusion(nn.Module):
    """
    Late Fusion Model based on Fig 2b.
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
        super(Attention_DC_CRN_LateFusion, self).__init__()
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
        
        out_len = out.shape[-1]
        
        if out_len > in_len:
            leftover = out_len - in_len 
            out = out[..., leftover//2:-(leftover//2)]
        
        return out
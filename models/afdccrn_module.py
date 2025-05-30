import torch
import torch.nn as nn
import torch.nn.functional as F

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

        Efficient Sequence Learning with Group Recurrent Networks; Gao et al., 2018 논문을 참고

        Args:
            hidden_size (int): grouped LSTM 레이어 내 모든 LSTM의 총 hidden size  
                → 각 LSTM의 hidden size = hidden_size // groups
            groups (int): 그룹의 수 (각 그룹별로 하나의 LSTM이 있음)
            layers (int): grouped LSTM 레이어의 수
            bidirectional (bool): 양방향 LSTM(BLSTM) 사용 여부  
            rearrange (bool): 각 grouped LSTM 레이어 후에 rearrange 연산 적용 여부
        """
        super().__init__()

        # hidden_size는 groups로 나누어 떨어져야 함
        assert hidden_size % groups == 0, (hidden_size, groups)
        # 각 그룹별 LSTM에 할당될 hidden size
        hidden_size_t = hidden_size // groups
        if bidirectional:
            # 양방향 LSTM인 경우 hidden_size_t는 2로 나누어 떨어져야 함
            assert hidden_size_t % 2 == 0, hidden_size_t

        self.groups = groups
        self.layers = layers
        self.rearrange = rearrange

        # 각 레이어별 LSTM과 LayerNorm을 저장하는 ModuleList 생성
        self.lstm_list = nn.ModuleList()
        # 레이어 수만큼 LayerNorm을 생성, 각 LayerNorm은 마지막 차원 hidden_size에 적용
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        
        # 각 grouped LSTM 레이어 구성
        for layer in range(layers):
            # 각 그룹에 대해 개별 LSTM을 생성해 ModuleList에 저장
            self.lstm_list.append(
                nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=hidden_size_t, 
                            # 양방향이면 hidden_size_t // 2, 단방향이면 hidden_size_t가 hidden dimension
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
            x (torch.Tensor): 입력 텐서, shape: (B, C, T, D)
        Returns:
            out (torch.Tensor): 출력 텐서, shape: (B, C, T, D)
        """
        # 1. 입력 x: (B, C, T, D)
        out = x

        # 2. C와 T 차원 교환 → shape: (B, T, C, D)
        out = out.transpose(1, 2).contiguous()

        # 3. (B, T, C, D)를 3D 텐서로 flatten  
        #    여기서 C와 D를 병합하여 shape: (B, T, C*D) 가 되어야 함.
        #    주의: C*D == hidden_size 이어야 함!
        B, T = out.size(0), out.size(1)
        out = out.view(B, T, -1).contiguous()  # shape: (B, T, hidden_size)

        # 4. 마지막 차원을 groups 개수로 분할 → 각 chunk shape: (B, T, hidden_size // groups)
        out = torch.chunk(out, self.groups, dim=-1)

        # 5. 첫 번째 grouped LSTM 레이어 처리  
        #    각 그룹별로 분할된 텐서를 해당 그룹의 LSTM에 넣어 처리  
        #    LSTM의 출력은 (B, T, hidden_size_t) (양방향의 경우에도 두 방향 합쳐서 hidden_size_t)
        #    리스트 컴프리헨션으로 각 그룹 결과를 받아 새 차원(dim=-1)에 쌓음
        out = torch.stack(
            [self.lstm_list[0][i](out[i])[0] for i in range(self.groups)], dim=-1
        )  # 결과 shape: (B, T, hidden_size_t, groups)

        # 6. 마지막 두 차원(flatten) → shape: (B, T, hidden_size_t * groups) = (B, T, hidden_size)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        # 7. 첫 번째 레이어에 대한 Layer Normalization 적용 → shape 유지: (B, T, hidden_size)
        out = self.ln[0](out)

        # 8. 두 번째 레이어부터 나머지 grouped LSTM 레이어 처리
        for layer in range(1, self.layers):
            # (옵션) rearrange 플래그가 True이면 그룹 순서를 재정렬
            if self.rearrange:
                # 현재 shape: (B, T, hidden_size)
                # reshape: (B, T, groups, hidden_size // groups)
                # transpose: 마지막 두 차원 교환 → (B, T, hidden_size // groups, groups)
                # 최종적으로 다시 flatten하여 (B, T, hidden_size)로 만듦
                out = (
                    out.reshape(B, T, self.groups, -1)
                    .transpose(-1, -2)
                    .contiguous()
                    .view(B, T, -1)
                )
            # 8-1. 현재 출력을 그룹 별로 분할  
            #     분할 후 각 텐서의 shape: (B, T, hidden_size // groups)
            out_chunks = torch.chunk(out, self.groups, dim=-1)

            # 8-2. 각 그룹별 LSTM 처리 → 각 출력 shape: (B, T, hidden_size_t)
            #       모든 그룹의 출력을 이어붙임(dim=-1) → (B, T, hidden_size)
            out = torch.cat(
                [self.lstm_list[layer][i](out_chunks[i])[0] for i in range(self.groups)],
                dim=-1,
            )
            # 8-3. 해당 레이어의 Layer Normalization 적용 → shape 유지: (B, T, hidden_size)
            out = self.ln[layer](out)

        # 9. 최종 출력 텐서 shape 변환  
        #    현재 shape: (B, T, hidden_size)
        #    원본 입력과 동일한 형식을 위해 다시 4D 텐서로 변환  
        #    x.size(1)는 원래의 C 차원, 따라서 D는 hidden_size / C가 됨
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()  # shape: (B, T, C, D)
        # 10. C와 T 차원 다시 교환 → 최종 출력 shape: (B, C, T, D)
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
            # print(f"Encoder {idx} input shape: {out.shape}")
            out = layer(out)
            # print(f"Encoder {idx} output shape: {out.shape}")
            skip = self.skip_pathway[idx](out)
            skips.insert(0, skip)
        
        out = out.permute(0, 1, 3, 2).contiguous()
        out = self.lstm(out)        
        out = out.permute(0, 1, 3, 2).contiguous()
        
        for idx in range(len(self.decoders)):
            skip = skips[idx]
            # print(f"Decoder {idx} skip shape: {skip.shape}")
            # print(f"Decoder {idx} out shape: {out.shape}")
            out = torch.cat((out, skip), dim=1)
            # print(f"Decoder {idx} input shape: {out.shape}")
            out = self.decoders[idx](out)
            # print(f"Decoder {idx} output shape: {out.shape}")

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

    
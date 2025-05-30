import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  1. 이전 단계에서 구현한 1D Involution 연산
# =============================================================================
class Involution1d(nn.Module):
    """
    논문에 설명된 1D Involution 연산 구현.
    원본 Involution (CVPR 2021) 아이디어를 1D 시계열 데이터에 적용.
    커널은 입력 피처에 따라 동적으로 생성됨.
    """
    def __init__(self,
                 in_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0, # Involution 자체는 보통 padding=0, Unfold에서 처리
                 dilation: int = 1,
                 groups: int = 1,
                 reduction_ratio: int = 4):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"'in_channels' ({in_channels}) must be divisible by 'groups' ({groups})")

        self.in_channels = in_channels
        self.out_channels = in_channels # Involution은 채널 수를 유지
        self.kernel_size = kernel_size
        self.stride = stride
        # Unfold에서 사용할 패딩 계산 (센터링 고려)
        self.padding = (kernel_size - 1) * dilation // 2
        self.dilation = dilation
        self.groups = groups
        self.reduction_ratio = reduction_ratio

        # 커널 생성 네트워크
        self.kernel_gen = nn.Sequential(
            # 논문 그림 2/4 구조에 따라 Involution 블록 전에 LayerNorm이 오므로,
            # 커널 생성 시에는 AveragePooling을 사용하지 않는 것이 더 적합할 수 있음.
            # 필요시 nn.AdaptiveAvgPool1d(1) if stride > 1 else nn.Identity(),
            nn.Conv1d(in_channels=self.in_channels,
                      out_channels=self.in_channels // self.reduction_ratio,
                      kernel_size=1),
            # nn.BatchNorm1d(self.in_channels // self.reduction_ratio),
            # nn.ReLU(),
            nn.Conv1d(in_channels=self.in_channels // self.reduction_ratio,
                      out_channels=self.groups * self.kernel_size,
                      kernel_size=1)
        )

        # Unfold 연산 정의 (nn.Unfold 사용)
        # nn.Unfold는 (B, C*K, L_out) 형태를 반환
        # Conv1d와 동작을 유사하게 맞추기 위해 padding을 여기서 직접 계산
        # 올바른 패딩 계산: (kernel_size - 1) * dilation // 2 (센터링 위함)
        self.unfold_padding = (0, (kernel_size - 1) * dilation // 2) # 왼쪽, 오른쪽 패딩
        # nn.Unfold 자체 패딩 대신 F.pad 사용 고려 -> 여기서는 nn.Unfold 패딩 사용
        self.unfold = nn.Unfold(kernel_size=(1, self.kernel_size),
                                dilation=(1, self.dilation),
                                padding=self.unfold_padding, # (H, W) 패딩
                                stride=(1, self.stride))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        # 입력 채널 수 검증 (초기화 시 이미 했지만 안전하게)
        assert channels == self.in_channels

        # 1. 커널 생성 (Input-dependent)
        # shape: (B, G*K, L) 또는 (B, G*K, L_out) <- stride > 1일 때 pooling 사용시
        kernel = self.kernel_gen(x)
        # 생성된 커널 shape: (B, G*K, L_kernel)
        # L_kernel = L / stride if pooling else L
        # 커널의 시퀀스 길이(L_kernel)는 이후 unfold 결과(L_out)와 일치해야 함
        # -> kernel_gen에서 stride를 고려하지 않았으므로 L_kernel = L

        # 2. 입력 피처 Unfold
        # F.unfold 또는 nn.Unfold 사용. nn.Unfold는 (B, C, H, W) 입력을 기대
        # 패딩 직접 추가: F.pad 사용 시
        # padded_x = F.pad(x, (self.padding, self.padding)) # (left, right)
        # x_unfolded = self.unfold(padded_x.unsqueeze(2)) # (B, C, 1, L_padded) -> (B, C*K, L_out)

        # nn.Unfold의 padding 인자 사용 시
        x_unfolded = self.unfold(x.unsqueeze(2)) # (B, C, 1, L) -> (B, C*K, L_out)

        # unfold 결과 shape: (B, C*K, L_out)
        # L_out = floor((L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        # 여기서는 padding = (K-1)*D//2 이므로, L_out = floor((L + (K-1)D - D*(K-1) - 1)/S + 1) ?? -> L_out = floor((L-1)/S + 1) ??
        # nn.Conv1d의 출력 길이와 동일하게 L_out = floor((L + 2*pad - D*(K-1) - 1) / S + 1)
        # 우리 padding = (K-1)*D//2 이므로, L_out = floor((L + (K-1)D - D*(K-1) - 1) / S + 1) = floor((L-1)/S + 1)
        # 만약 Stride=1이면 L_out = L
        _, Ck, L_out = x_unfolded.shape

        # 3. Reshape for Multiply-Add
        # 커널 shape: (B, G*K, L_out) -> (B, G, K, L_out)
        kernel = kernel.view(batch_size, self.groups, self.kernel_size, L_out)
        # Unfolded 피처 shape: (B, C*K, L_out) -> (B, G, C//G, K, L_out)
        x_unfolded = x_unfolded.view(batch_size, self.groups, channels // self.groups, self.kernel_size, L_out)

        # 4. Multiply-Add 연산
        # 커널 브로드캐스팅: (B, G, 1, K, L_out) * (B, G, C//G, K, L_out) -> (B, G, C//G, K, L_out)
        out_unfolded = kernel.unsqueeze(2) * x_unfolded
        # 커널 차원(K) 합산: (B, G, C//G, L_out)
        out = out_unfolded.sum(dim=3)

        # 5. Final Reshape
        # (B, G, C//G, L_out) -> (B, C, L_out)
        out = out.view(batch_size, channels, L_out) # 채널 수는 입력과 동일 (out_channels=in_channels)

        return out


# =============================================================================
#  2. 논문 그림 4의 1D Involution Block 구현
# =============================================================================
class InvolutionBlock(nn.Module):
    """
    논문 그림 4의 1D Involution Block.
    LayerNorm -> ( P x (Involution1d -> PReLU) ) -> + -> Output
                      |                               |
                      ------> (1x1 Conv) -------------
    """
    def __init__(self,
                 channels: int,          # 입력 및 출력 채널 수 (N)
                 kernel_size: int,       # Involution 커널 크기 (K)
                 groups: int,            # Involution 그룹 수 (G)
                 reduction_ratio: int,   # Involution 리덕션 비율
                 num_inv_blocks: int):   # 반복할 Involution1d + PReLU 개수 (P)
        super().__init__()
        self.channels = channels
        self.num_inv_blocks = num_inv_blocks

        # Layer Normalization (그림 4에서 Involution 연산 전에 적용)
        self.layer_norm = nn.LayerNorm(normalized_shape=channels) # 채널 방향으로 LN
        self.act = nn.PReLU()

        # Main Path: P번 반복되는 Involution + PReLU
        self.main_path = nn.ModuleList()
        for _ in range(num_inv_blocks):
            self.main_path.append(
                Involution1d(
                    in_channels=channels,
                    kernel_size=kernel_size,
                    stride=1, # 블록 내에서는 stride=1
                    groups=groups,
                    reduction_ratio=reduction_ratio
                )
            )
            self.main_path.append(nn.PReLU(num_parameters=channels)) # 채널별 PReLU 파라미터

        # Skip Connection Path: 1x1 Convolution
        self.skip_path = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. Shape: (batch_size, channels, sequence_length)
        Returns:
            torch.Tensor: 블록 출력 텐서. Shape: (batch_size, channels, sequence_length)
        """
        # Layer Normalization 적용 전에 (B, C, L) -> (B, L, C) 형태로 변경 필요
        residual = x
        x = self.act(x)
        x_norm = self.layer_norm(x.transpose(1, 2)).transpose(1, 2) # LN -> (B, L, C) -> LN -> (B, C, L)

        # Main Path 연산
        main_out = x_norm
        for layer in self.main_path:
            main_out = layer(main_out)

        # Skip Path 연산
        skip_out = self.skip_path(x_norm) # Norm 적용된 x 사용

        # Residual Connection (Main Path + Skip Path)
        output = main_out + skip_out

        # 논문 그림에서는 블록 결과에 residual connection (입력 x) 를 더하는지는 명확하지 않음.
        # 그림 4만 보면 main+skip 이 최종 출력. 만약 필요하다면 output = output + residual 추가.
        # 여기서는 그림 4 구조대로 main + skip 을 반환.

        return output


# =============================================================================
#  3. Encoder (1D Convolution)
# =============================================================================
class Encoder(nn.Module):
    """ 논문 2.1절의 Encoder 구현 """
    def __init__(self, kernel_size: int = 16, num_channels: int = 256):
        """
        Args:
            kernel_size (int): Conv1d 커널 크기 (L). 기본값 16.
            num_channels (int): Conv1d 출력 채널 수 (N). 기본값 256.
        """
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,           # 입력은 단일 채널 시계열 데이터
            out_channels=num_channels, # 출력 채널 N
            kernel_size=kernel_size,   # 커널 크기 L
            stride=kernel_size // 2, # 스트라이드 L/2
            padding=0,               # 논문에 패딩 언급 없음, 0으로 설정
            bias=False               # Conv-TasNet 스타일 따라 bias=False 사용 가능
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 파형. Shape: (batch_size, sequence_length)
        Returns:
            torch.Tensor: 인코딩된 피처. Shape: (batch_size, num_channels, feature_length)
        """
        # (B, L_in) -> (B, 1, L_in)
        x = x.unsqueeze(1)
        print(f"x.shape: {x.shape}")
        # (B, 1, L_in) -> (B, N, L_feat)
        encoded = self.conv1d(x)
        print(f"encoded.shape: {encoded.shape}")
        return encoded

# =============================================================================
#  4. Decoder (1D Transposed Convolution)
# =============================================================================
class Decoder(nn.Module):
    """ 논문 2.3절의 Decoder 구현 """
    def __init__(self, kernel_size: int = 16, num_channels: int = 256):
        """
        Args:
            kernel_size (int): ConvTranspose1d 커널 크기 (L). 기본값 16.
            num_channels (int): ConvTranspose1d 입력 채널 수 (N). 기본값 256.
        """
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels=num_channels,  # 입력 채널 N
            out_channels=1,            # 출력은 단일 채널 시계열 데이터
            kernel_size=kernel_size,   # 커널 크기 L
            stride=kernel_size // 2, # 스트라이드 L/2
            padding=0,               # 논문에 패딩 언급 없음, 0으로 설정
                                     # 필요시 output_padding 조절하여 길이 맞춤
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 디코더 입력 피처. Shape: (batch_size, num_channels, feature_length)
        Returns:
            torch.Tensor: 재구성된 파형. Shape: (batch_size, sequence_length)
        """
        # (B, N, L_feat) -> (B, 1, L_out)
        decoded = self.conv_transpose1d(x)
        # (B, 1, L_out) -> (B, L_out)
        return decoded.squeeze(1)


# =============================================================================
#  5. Mask Estimator (논문 그림 2, 4 기반)
# =============================================================================
class MaskEstimator(nn.Module):
    """
    논문 2.2절의 Mask Estimator 구현 (그림 2).
    B개의 InvolutionBlock 스택으로 구성됨.
    """
    def __init__(self,
                 input_channels: int,    # 인코더 출력 채널 합 (2N)
                 block_channels: int,    # Involution 블록 내부 채널 (N)
                 num_blocks: int,        # 쌓을 InvolutionBlock 개수 (B)
                 inv_kernel_size: int,   # Involution 커널 크기 (K)
                 inv_groups: int,        # Involution 그룹 수 (G)
                 inv_reduction_ratio: int, # Involution 리덕션 비율
                 num_inv_per_block: int): # 블록 당 Involution 반복 수 (P)
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=input_channels) # 채널 방향으로 LN
        # 초기 1x1 Conv: 입력 채널(2N)을 블록 채널(N)으로 조정
        self.initial_conv = nn.Conv1d(in_channels=input_channels,
                                       out_channels=block_channels,
                                       kernel_size=1)

        # B개의 InvolutionBlock 스택
        self.inv_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.inv_blocks.append(
                InvolutionBlock(
                    channels=block_channels,
                    kernel_size=inv_kernel_size,
                    groups=inv_groups,
                    reduction_ratio=inv_reduction_ratio,
                    num_inv_blocks=num_inv_per_block
                )
            )

        # 최종 마스크 생성 레이어: 1x1 Conv + Activation
        # 출력 채널은 AC 인코더 채널 수(N)와 같아야 함 [source: 320]
        self.final_conv = nn.Conv1d(in_channels=block_channels,
                                     out_channels=block_channels, # Mask 채널 = N
                                     kernel_size=1)
        # 논문에서 마스크 활성화 함수 명시 안됨. Sigmoid가 일반적이나, ReLU 사용 가능성도 있음.
        # 여기서는 Sigmoid 사용 (0~1 범위 마스크 생성)
        # self.mask_activation = nn.Sigmoid()
        self.mask_activation = nn.ReLU() # 또는 ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 인코딩된 AC와 BC 피처가 concat된 텐서. Shape: (B, N, L_feat)
        Returns:
            torch.Tensor: 추정된 마스크. Shape: (B, N, L_feat)
        """
        x = x.transpose(1, 2) # (B, N, L_feat) -> (B, L_feat, N)
        # Layer Normalization 적용 (그림 4에서 Involution 연산 전에 적용)       
        x = self.layer_norm(x)
        x = x.transpose(1, 2) # (B, L_feat, N) -> (B, N, L_feat)
        # 초기 1x1 Conv: (B, N, L_feat) -> (B, N, L_feat)
        x = self.initial_conv(x)

        # Involution Blocks 통과
        for block in self.inv_blocks:
            x = block(x) # 각 블록은 (B, N, L_feat) -> (B, N, L_feat)

        # 최종 마스크 생성: (B, N, L_feat) -> (B, N, L_feat)
        mask = self.final_conv(x)
        mask = self.mask_activation(mask)
        return mask

# =============================================================================
#  6. MMINet 전체 모델 (논문 그림 1 기반)
# =============================================================================
class MMINet(nn.Module):
    """
    Multi-modal Involution Network (MMINet) 구현.
    논문 "Multi-modal speech enhancement with bone-conducted speech in time domain" 기반.
    """
    def __init__(self,
                 L: int = 16,   # Encoder/Decoder 커널 크기
                 N: int = 256,   # Encoder/Decoder 채널 수
                 B: int = 4,    # Mask Estimator의 InvolutionBlock 스택 수
                 P: int = 3,    # InvolutionBlock 당 Involution 반복 수
                 K: int = 3,    # Involution 커널 크기
                 G: int = 1,    # Involution 그룹 수
                 reduction_ratio: int = 4): # Involution 리덕션 비율
        """
        Args:
            L (int): Encoder/Decoder Conv1d 커널 크기 [source: 292]. Default: 16.
            N (int): Encoder/Decoder Conv1d 채널 수 [source: 292]. Default: 64.
            B (int): Mask Estimator 내 InvolutionBlock 스택 수 [source: 314]. Default: 4.
            P (int): InvolutionBlock 내 Involution 반복 수 [source: 314]. Default: 3.
            K (int): Involution 커널 크기 [source: 314]. Default: 5.
            G (int): Involution 그룹 수 [source: 314]. Default: 8.
            reduction_ratio (int): Involution 커널 생성 시 리덕션 비율. Default: 4.
        """
        super().__init__()
        self.L = L
        self.N = N
        self.B = B
        self.P = P
        self.K = K
        self.G = G
        self.reduction_ratio = reduction_ratio

        # Speech Encoder
        self.ac_encoder = Encoder(kernel_size=L, num_channels=N) # AC 인코더
        self.bc_encoder = Encoder(kernel_size=L, num_channels=N) # BC 인코더

        # Mask Estimator
        self.mask_estimator = MaskEstimator(
            input_channels=N,      # AC(N) + BC(N) 채널 결합
            block_channels=N,          # 블록 내부 채널 수
            num_blocks=B,              # 블록 스택 수
            inv_kernel_size=K,         # Involution 커널 크기
            inv_groups=G,              # Involution 그룹 수
            inv_reduction_ratio=reduction_ratio,
            num_inv_per_block=P        # 블록 당 Involution 반복 수
        )

        # Decoder
        self.decoder = Decoder(kernel_size=L, num_channels=N)

    def forward(self, noisy_ac: torch.Tensor, noisy_bc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_ac (torch.Tensor): 노이즈 포함된 공기 전도 음성 파형. Shape: (batch_size, ac_len)
            noisy_bc (torch.Tensor): 노이즈 포함된 (또는 깨끗한) 골전도 음성 파형. Shape: (batch_size, bc_len)
                                     ac_len과 bc_len은 동일해야 함.
        Returns:
            torch.Tensor: 향상된 공기 전도 음성 파형. Shape: (batch_size, enhanced_len)
        """
        
        # 1. Encoding [source: 278, Fig 1]
        # (B, L_in) -> (B, 1, L_in)
        
        # AC 인코더: (B, 1, L_in) -> (B, N, L_feat)
        ac_encoded = self.ac_encoder(noisy_ac)
        # BC 인코더: (B, 1, L_in) -> (B, N, L_feat)
        bc_encoded = self.bc_encoder(noisy_bc)
        
        # 2. Feature Concatenation [source: 297]
        encoded = ac_encoded + bc_encoded

        # 3. Mask Estimation [source: 278, Fig 1]
        # (B, N, L_feat) -> (B, N, L_feat)
        estimated_mask = self.mask_estimator(encoded)

        # 4. Mask Application [source: 320]
        # 마스크는 AC 인코더 피처에 적용됨
        # (B, N, L_feat) * (B, N, L_feat) -> (B, N, L_feat)
        masked_ac_features = encoded * estimated_mask

        # 5. Decoding [source: 278, Fig 1]
        # (B, N, L_feat) -> (B, L_out)
        enhanced_ac = self.decoder(masked_ac_features)

        # Decoder 출력 길이는 Encoder 입력 길이와 정확히 일치하지 않을 수 있음.
        # ConvTranspose1d의 stride, padding, output_padding 파라미터에 따라 달라짐.
        # 필요시 원본 길이에 맞게 잘라내거나 패딩 추가.
        original_len = noisy_ac.shape[-1]
        output_len = enhanced_ac.shape[-1]
        if output_len > original_len:
            enhanced_ac = enhanced_ac[..., :original_len]
        elif output_len < original_len:
            # 필요한 경우 패딩 (보통은 약간 길게 나옴)
             padding_size = original_len - output_len
             enhanced_ac = F.pad(enhanced_ac, (0, padding_size))


        return enhanced_ac
import torch
import torch.nn as nn

'''
Unofficial implementation of the Early Fusion FCN model (FCN_EF) from the paper:
"Time-Domain Multi-modal Bone/air Conducted Speech Enhancement", Yu et al. 2020.
'''
class FCN_EarlyFusion(nn.Module):
    """ Early Fusion Fully Convolutional Network (FCN_EF) """
    def __init__(self,
                 depth: int = 6,
                 kernel_size: int = 55,
                 channels: int = 30,
                 ):
        super().__init__()
        
        self.depth = depth
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.channel = channels    
        
        layers = []    
        for i in range (depth):
            layers.append(
                nn.Conv1d(channels if i > 0 else 2, channels, kernel_size=self.kernel_size, padding=self.padding)
            )
            layers.append(nn.ReLU())
        
        # Last layer to reduce to 1 channel
        layers.append(nn.Conv1d(channels, 1, kernel_size=self.kernel_size, padding=self.padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_bcm: torch.Tensor, x_acm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_bcm, x_acm], dim=1)  # Concatenate along channel dimension
        x = self.layers(x)  # Pass through the network
        return x
    

class FCN_LateFusion(nn.Module):
    """ Late Fusion Fully Convolutional Network (FCN_LF) """
    def __init__(self,
                 a_depth: int = 6,
                 a_channels: int = 33,
                 a_kernel_size: int = 55,
                 b_channels: list = [1, 3, 5, 1],
                 b_kernel_size: list = [257, 1, 15, 513],
                 lf_channels: int = 15,
                 lf_kernel_size: int = 55,
                 ):
        super().__init__()
        
        self.a_depth = a_depth
        self.a_channelss = a_channels
        self.a_kernel_size = a_kernel_size
        self.b_channels = b_channels
        self.b_kernel_size = b_kernel_size
        self.lf_channels = lf_channels
        self.lf_kernel_size = lf_kernel_size
        
        fcn_a = []
        for i in range(a_depth):
            fcn_a.append(
                nn.Conv1d(
                    1 if i == 0 else a_channels,
                    a_channels,
                    kernel_size=a_kernel_size,
                    padding=(a_kernel_size - 1) // 2
                )
            )
            fcn_a.append(nn.ReLU())
        
        fcn_a.append(
            nn.Conv1d(a_channels, 1, kernel_size=a_kernel_size, padding=(a_kernel_size - 1) // 2)
        )
        self.fcn_a = nn.Sequential(*fcn_a)
        
        fcn_b = []
        channel = 1
        for i in range(len(b_channels)):
            fcn_b.append(
                nn.Conv1d(
                    channel,
                    b_channels[i],
                    kernel_size=b_kernel_size[i],
                    padding=(b_kernel_size[i] - 1) // 2                    
                )
            )
            fcn_b.append(nn.ReLU())
            channel = b_channels[i]
            
        self.fcn_b = nn.Sequential(*fcn_b)
                
        # Fusion network (FCN_LF)
        self.fcn_lf = nn.Sequential(
            nn.Conv1d(2, lf_channels, kernel_size=lf_kernel_size, padding=(lf_kernel_size - 1) // 2),  # Input: concatenated outputs
            nn.ReLU(),
            nn.Conv1d(lf_channels, 1, kernel_size=lf_kernel_size, padding=(lf_kernel_size - 1) // 2),  # Output: single channel
        )

    def forward(self, x_bcm: torch.Tensor, x_acm: torch.Tensor) -> torch.Tensor:
        s_b = self.fcn_b(x_bcm)          # Process BCM signal
        s_a = self.fcn_a(x_acm)          # Process ACM signal
        s_concat = torch.cat([s_b, s_a], dim=1)  # Concatenate outputs
        return self.fcn_lf(s_concat)     # Final enhancement
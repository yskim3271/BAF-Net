import torch
import torch.nn as nn


'''
Unofficial implementation of the Early Fusion FCN model (FCN_EF) from the paper:
"Time-Domain Multi-modal Bone/air Conducted Speech Enhancement", Yu et al. 2020.
'''
class fcnef(nn.Module):
    """ Early Fusion Fully Convolutional Network (FCN_EF) """
    def __init__(self,
                 depth: int = 6,
                 kernel_size: int = 55,
                 channel: int = 30,
                 ):
        super().__init__()
        
        self.depth = depth
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.channel = channel    
        
        layers = []    
        for i in range (depth):
            layers.append(
                nn.Conv1d(channel if i > 0 else 2, channel, kernel_size=self.kernel_size, padding=self.padding)
            )
            layers.append(nn.ReLU())
        
        # Last layer to reduce to 1 channel
        layers.append(nn.Conv1d(channel, 1, kernel_size=self.kernel_size, padding=self.padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_bcm: torch.Tensor, x_acm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_bcm, x_acm], dim=1)  # Concatenate along channel dimension
        x = self.layers(x)  # Pass through the network
        return x
import torch
import torch.nn as nn

from models.afdccrn_module import AttentionFusion, DCBlock, GLSTM
from models.mminet import MMINet
from models.afdccrn_ef import afdccrn_ef
from models.afdccrn_lf import afdccrn_lf

def test_attention_fusion():
    # Create a sample input tensor
    x = torch.randn(1, 3, 11, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions
    y = torch.randn(1, 3, 64, 64)  # Another sample input tensor

    # Create an instance of the AttentionFusion module
    attention_fusion = AttentionFusion(channels=3)

    # Forward pass through the module
    output = attention_fusion(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("AttentionFusion module test passed!")
    

def test_mminet():
    # Create a sample input tensor
    x1 = torch.randn(1, 16000)  # Batch size of 1, 1 channel, 16000 samples
    x2 = torch.randn(1, 16000)  # Another sample input tensor
    # Create an instance of the MMINet module
    mmi_net = MMINet()

    # Forward pass through the module
    output = mmi_net(x1, x2)

def test_dcblock():
    # Create a sample input tensor
    x = torch.randn(1, 2, 256, 126)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

    # Create an instance of the DCBlock module
    dcblock = DCBlock(in_channels=2, out_channels=4, encode=True)

    # Forward pass through the module
    output = dcblock(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


def test_afdccrn_ef():
    # Create a sample input tensor
    x1 = torch.randn(1, 32000)
    x2 = torch.randn(1, 32000)
    # Create an instance of the afdccrn_ef module
    afdccrn = afdccrn_ef()

    # Forward pass through the module
    output = afdccrn(x1, x2)

    print("Output shape:", output.shape)


def test_glstm():
    x = torch.rand(1, 256, 4, 126)
    
    
    lstm = GLSTM(hidden_size=1024, groups=2, layers=2, bidirectional=True)
    
    x = x.permute(0, 1, 3, 2).contiguous()
    print(f"Input shape: {x.shape}")
    
    output = lstm(x)
    print("Output shape:", output.shape)

def test_afdccrn_lf():
    # Create a sample input tensor
    x1 = torch.randn(1, 32000)
    x2 = torch.randn(1, 32000)
    # Create an instance of the afdccrn_ef module
    afdccrn = afdccrn_ef()

    # Forward pass through the module
    output = afdccrn(x1, x2)

    print("Output shape:", output.shape)
    
    

if __name__ == "__main__":
    # test_attention_fusion()
    # test_dcblock()
    # test_mminet()
    # test_afdccrn_ef()
    # test_glstm()
    test_afdccrn_lf()
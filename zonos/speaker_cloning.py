"""
Speaker cloning and embedding modules for the Zonos TTS system.

This module provides neural network components for speaker verification and embedding
extraction, supporting voice cloning capabilities in text-to-speech synthesis.

Key Components:
    - ResNet293-based models with SimAM attention for robust speaker embeddings
    - ECAPA-TDNN architecture for time-delay neural networks with channel attention
    - LDA-projected embeddings for dimensionality reduction and improved discrimination
    - Log filterbank feature extraction for audio preprocessing

Architecture Overview:
    The speaker cloning system uses ResNet-based deep networks with specialized attention
    mechanisms (SimAM) to extract speaker-specific embeddings from audio. These embeddings
    are used in the Zonos TTS model's prefix conditioning system to control voice characteristics
    during speech synthesis.

Integration:
    - audio_utils.py: make_speaker_embedding() uses SpeakerEmbeddingLDA for feature extraction
    - test_zonos.py: process_speaker_audio() integrates speaker embeddings into TTS pipeline
    - model.py: speaker embeddings flow through prefix_conditioning in generate() method

Models trained on VoxCeleb dataset for speaker verification tasks.
"""

import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

from zonos.utilities.utils import DEFAULT_DEVICE


class logFbankCal(nn.Module):
    """
    Log filterbank feature calculator for audio preprocessing.
    
    Computes log mel-scale filterbank features from raw audio waveforms, commonly used
    in speaker recognition systems. Applies mean normalization for robust feature extraction.
    
    Args:
        sample_rate (int, optional): Audio sampling rate in Hz. Defaults to 16000.
        n_fft (int, optional): FFT size for spectrogram computation. Defaults to 512.
        win_length (float, optional): Window length in seconds. Defaults to 0.025.
        hop_length (float, optional): Hop length in seconds. Defaults to 0.01.
        n_mels (int, optional): Number of mel filterbank channels. Defaults to 80.
        
    Input:
        x (torch.Tensor): Raw audio waveform of shape [batch_size, num_samples]
        
    Output:
        torch.Tensor: Log mel filterbank features of shape [batch_size, n_mels, time_frames]
                     with mean normalization applied across time dimension
                     
    Note:
        Features are normalized by subtracting the temporal mean, improving robustness
        to channel variations and recording conditions.
    """
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: float = 0.025,
        hop_length: float = 0.01,
        n_mels: int = 80,
    ):
        super().__init__()
        self.fbankCal = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=int(win_length * sample_rate),
            hop_length=int(hop_length * sample_rate),
            n_mels=n_mels,
        )
        # Pre-compute for efficiency
        self.register_buffer('eps', torch.tensor(1e-6))

    def forward(self, x):
        out = self.fbankCal(x)
        # More memory-efficient computation
        out = torch.log(out + self.eps)
        # One-line mean subtraction
        mean = out.mean(dim=2, keepdim=True)
        return out - mean


class ASP(nn.Module):
    """
    Attentive Statistics Pooling (ASP) module for temporal aggregation.
    
    Computes weighted statistics (mean and standard deviation) across the temporal
    dimension using learned attention weights. Commonly used in speaker recognition
    to create fixed-size embeddings from variable-length sequences.
    
    Args:
        in_planes (int): Number of input channels from backbone network
        acoustic_dim (int): Acoustic feature dimension (typically 80 for mel features)
        
    Input:
        x (torch.Tensor): Feature maps of shape [batch_size, channels, height, width]
                         where channels = in_planes * 8 * (acoustic_dim // 8)
                         
    Output:
        torch.Tensor: Aggregated features of shape [batch_size, out_dim]
                     where out_dim = in_planes * 8 * (acoustic_dim // 8) * 2
                     
    Note:
        The factor of 2 comes from concatenating both mean (mu) and standard deviation (sg).
        Standard deviation is clamped to prevent numerical instability.
    """
    # Attentive statistics pooling
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2

        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        return x


class SimAMBasicBlock(nn.Module):
    """
    ResNet basic block with SimAM (Simple, parameter-free Attention Module).
    
    Implements a residual block enhanced with SimAM attention mechanism, which provides
    spatial and channel attention without additional parameters. SimAM computes attention
    weights based on the energy function that measures the importance of each neuron.
    
    Args:
        ConvLayer (nn.Module): Convolution layer class (nn.Conv1d, nn.Conv2d, etc.)
        NormLayer (nn.Module): Normalization layer class (nn.BatchNorm1d, nn.BatchNorm2d, etc.)
        in_planes (int): Number of input channels
        planes (int): Number of output channels
        stride (int, optional): Convolution stride. Defaults to 1.
        block_id (int, optional): Block identifier for debugging. Defaults to 1.
        
    Input:
        x (torch.Tensor): Input feature tensor
        
    Output:
        torch.Tensor: Output feature tensor with same spatial dimensions (if stride=1)
                     or downsampled by stride factor
                     
    Note:
        SimAM attention uses an energy function E = 1/t * Σ(xi - μ)² where t is the
        spatial size, providing parameter-free attention mechanism.
    """
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        """
        Simple, parameter-free Attention Module (SimAM) implementation.
        
        Computes attention weights based on energy minimization without additional parameters.
        The attention mechanism enhances important features while suppressing less relevant ones.
        
        Args:
            X (torch.Tensor): Input feature tensor of shape [batch_size, channels, height, width]
            lambda_p (float, optional): Regularization parameter to prevent division by zero.
                                       Defaults to 1e-4.
                                       
        Returns:
            torch.Tensor: Attention-weighted features with same shape as input
            
        Mathematical Formula:
            E_inv = d / (4 * (v + λ)) + 0.5
            where d = (X - μ)², v = Σd / n, n = H*W - 1, μ = mean(X)
        """
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block without attention mechanism.
    
    Implements the fundamental residual building block with two 3x3 convolutions,
    batch normalization, and ReLU activation. Used as a baseline comparison
    to the SimAM-enhanced blocks.
    
    Args:
        ConvLayer (nn.Module): Convolution layer class (nn.Conv1d, nn.Conv2d, etc.)
        NormLayer (nn.Module): Normalization layer class (nn.BatchNorm1d, nn.BatchNorm2d, etc.)
        in_planes (int): Number of input channels
        planes (int): Number of output channels
        stride (int, optional): Convolution stride. Defaults to 1.
        block_id (int, optional): Block identifier for debugging. Defaults to 1.
        
    Input:
        x (torch.Tensor): Input feature tensor
        
    Output:
        torch.Tensor: Output feature tensor with residual connection
        
    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
        |                                              ↑
        └─────── downsample (if needed) ──────────────┘
    """
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    ResNet bottleneck block for deeper networks (ResNet-50+).
    
    Implements the 1x1 -> 3x3 -> 1x1 convolution pattern that reduces computational
    cost while maintaining representational capacity. The expansion factor of 4
    increases the channel dimension in the final 1x1 convolution.
    
    Args:
        ConvLayer (nn.Module): Convolution layer class (typically nn.Conv2d)
        NormLayer (nn.Module): Normalization layer class (typically nn.BatchNorm2d)
        in_planes (int): Number of input channels
        planes (int): Number of intermediate channels (bottleneck width)
        stride (int, optional): Stride for the 3x3 convolution. Defaults to 1.
        block_id (int, optional): Block identifier for debugging. Defaults to 1.
        
    Input:
        x (torch.Tensor): Input feature tensor
        
    Output:
        torch.Tensor: Output feature tensor with channels = planes * expansion (4)
        
    Note:
        This implementation is fixed to 2D convolutions and may not be used
        in the current speaker embedding models which primarily use 1D/2D basic blocks.
    """
    expansion = 4

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Generic ResNet backbone architecture supporting 1D, 2D, and 3D convolutions.
    
    Flexible ResNet implementation that can operate on different dimensional data
    (1D for temporal sequences, 2D for spectrograms, 3D for video-like data).
    Used as the backbone for speaker embedding extraction.
    
    Args:
        in_planes (int): Number of channels in each layer (base width)
        block (nn.Module): Residual block class (BasicBlock, SimAMBasicBlock, or Bottleneck)
        num_blocks (List[int]): Number of blocks in each of the 4 layers
        in_ch (int, optional): Number of input channels. Defaults to 1.
        feat_dim (str, optional): Feature dimension type ("1d", "2d", or "3d"). Defaults to "2d".
        **kwargs: Additional arguments passed to the block constructor
        
    Input:
        x (torch.Tensor): Input tensor with shape depending on feat_dim:
                         - 1D: [batch_size, in_ch, sequence_length]
                         - 2D: [batch_size, in_ch, height, width]
                         - 3D: [batch_size, in_ch, depth, height, width]
                         
    Output:
        torch.Tensor: Feature maps from the final layer (layer4)
        
    Architecture:
        Input -> Conv -> BN -> ReLU -> Layer1 -> Layer2 -> Layer3 -> Layer4
        Each layer contains num_blocks[i] residual blocks with progressively
        doubling channels: in_planes -> 2*in_planes -> 4*in_planes -> 8*in_planes
    """
    def __init__(self, in_planes, block, num_blocks, in_ch=1, feat_dim="2d", **kwargs):
        super(ResNet, self).__init__()
        if feat_dim == "1d":
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        elif feat_dim == "2d":
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d
        elif feat_dim == "3d":
            self.NormLayer = nn.BatchNorm3d
            self.ConvLayer = nn.Conv3d
        else:
            print("error")

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, block_id=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, block_id=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, block_id=3)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, block_id=4)

    def _make_layer(self, block, planes, num_blocks, stride, block_id=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride, block_id))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def ResNet293(in_planes: int, **kwargs):
    """
    ResNet293 architecture factory function.
    
    Creates a ResNet with 293 total layers using SimAMBasicBlock with attention.
    The architecture uses [10, 20, 64, 3] blocks in the four layers, specifically
    designed for speaker embedding tasks.
    
    Args:
        in_planes (int): Number of channels in the first layer (base width)
        **kwargs: Additional arguments passed to the ResNet constructor
        
    Returns:
        ResNet: ResNet293 model instance with SimAM attention blocks
        
    Architecture Details:
        - Layer 1: 10 SimAMBasicBlocks, channels = in_planes
        - Layer 2: 20 SimAMBasicBlocks, channels = 2 * in_planes, stride=2
        - Layer 3: 64 SimAMBasicBlocks, channels = 4 * in_planes, stride=2  
        - Layer 4: 3 SimAMBasicBlocks, channels = 8 * in_planes, stride=2
        Total: 2 + 2*(10+20+64+3) = 2 + 194 = 196 conv layers + other layers ≈ 293 total
    """
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
    """
    Complete speaker embedding model based on ResNet293 architecture.
    
    Combines ResNet293 backbone with log filterbank feature extraction, attentive
    statistics pooling, and embedding projection. This is the primary model for
    extracting speaker embeddings from raw audio.
    
    Args:
        in_planes (int, optional): Base channel width for ResNet layers. Defaults to 64.
        embd_dim (int, optional): Final embedding dimension. Defaults to 256.
        acoustic_dim (int, optional): Input acoustic feature dimension. Defaults to 80.
        featCal (nn.Module, optional): Feature calculator module. If None, uses logFbankCal.
        dropout (float, optional): Dropout rate before final projection. Defaults to 0.
        **kwargs: Additional arguments passed to ResNet293
        
    Input:
        x (torch.Tensor): Raw audio waveform of shape [batch_size, num_samples]
        
    Output:
        torch.Tensor: Speaker embedding of shape [batch_size, embd_dim]
        
    Pipeline:
        Audio -> logFbankCal -> ResNet293 -> ASP -> [Dropout] -> Linear -> Embedding
        
    Note:
        Used by SpeakerEmbedding class for extracting 256-dimensional speaker embeddings
        from 16kHz audio input.
    """
    def __init__(
        self,
        in_planes: int = 64,
        embd_dim: int = 256,
        acoustic_dim: int = 80,
        featCal=None,
        dropout: float = 0,
        **kwargs,
    ):
        super(ResNet293_based, self).__init__()
        self.featCal = featCal
        self.front = ResNet293(in_planes)
        block_expansion = SimAMBasicBlock.expansion
        self.pooling = ASP(in_planes * block_expansion, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation (SE) module for channel attention.
    
    Implements channel-wise attention mechanism that adaptively recalibrates
    channel-wise feature responses by explicitly modeling interdependencies
    between channels.
    
    Args:
        channels (int): Number of input channels
        bottleneck (int, optional): Bottleneck dimension for compression. Defaults to 128.
        
    Input:
        input (torch.Tensor): Input feature tensor of shape [batch_size, channels, length]
        
    Output:
        torch.Tensor: Channel-wise attended features with same shape as input
        
    Architecture:
        Input -> GlobalAvgPool -> Conv1x1(bottleneck) -> ReLU -> Conv1x1(channels) -> Sigmoid -> Scale
        
    Note:
        The bottleneck compression reduces parameters and computational cost while
        maintaining the modeling capacity for channel relationships.
    """
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # Removed
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    """
    Bottle2neck residual block with multi-scale feature processing.
    
    Implements the Res2Net building block that represents multi-scale features at
    a granular level within a single residual block. Each subset of channels is
    processed with different scales of receptive fields.
    
    Args:
        inplanes (int): Number of input channels
        planes (int): Number of output channels
        kernel_size (int, optional): Kernel size for dilated convolutions
        dilation (int, optional): Dilation rate for dilated convolutions
        scale (int, optional): Number of scales (subsets). Defaults to 8.
        
    Input:
        x (torch.Tensor): Input feature tensor of shape [batch_size, inplanes, length]
        
    Output:
        torch.Tensor: Multi-scale processed features of shape [batch_size, planes, length]
        
    Architecture:
        Uses hierarchical residual connections where each subset processes features
        at different temporal scales through dilated convolutions, enabling the model
        to capture both fine-grained and coarse-grained temporal patterns.
        
    Note:
        Essential component of ECAPA-TDNN architecture, providing multi-resolution
        temporal modeling capabilities for robust speaker embeddings.
    """
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network).
    
    State-of-the-art speaker embedding architecture that combines time-delay neural networks
    with channel attention and multi-scale feature aggregation. Designed for robust speaker
    verification and identification tasks.
    
    Args:
        C (int): Base channel dimension for the network
        featCal (nn.Module): Feature calculator for input preprocessing (typically logFbankCal)
        
    Input:
        x (torch.Tensor): Raw audio waveform of shape [batch_size, num_samples]
        
    Output:
        torch.Tensor: Speaker embedding of shape [batch_size, 192]
        
    Architecture Components:
        1. Feature extraction: Converts audio to log mel-scale features (80-dim)
        2. Initial 1D conv: Projects features to C channels with 5x5 kernel
        3. Multi-scale layers: 3 Bottle2neck blocks with increasing dilation (2,3,4)
        4. Feature aggregation: Concatenates multi-scale features -> 1536 channels
        5. Attention pooling: Global context attention for temporal aggregation
        6. Final projection: BatchNorm -> Linear -> BatchNorm -> 192-dim embedding
        
    Key Features:
        - Multi-scale temporal processing through dilated convolutions
        - Channel attention via SE modules in Bottle2neck blocks  
        - Global context attention for temporal aggregation
        - Designed for variable-length audio input
        
    Note:
        Based on "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
        in TDNN Based Speaker Verification" (Desplanques et al., 2020)
    """
    def __init__(self, C, featCal):
        super(ECAPA_TDNN, self).__init__()
        self.featCal = featCal
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # Added
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        x = self.featCal(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


class SpeakerEmbedding(nn.Module):
    """
    High-level speaker embedding extraction model using ResNet293.
    
    Provides a convenient interface for extracting speaker embeddings from raw audio
    using a pre-trained ResNet293-based model. Handles device management, audio 
    preprocessing, and model inference.
    
    Args:
        ckpt_path (str, optional): Path to pre-trained model checkpoint.
                                  Defaults to "ResNet293_SimAM_ASP_base.pt".
        device (torch.device, optional): Target device for computation. 
                                        Defaults to DEFAULT_DEVICE.
                                        
    Input (via forward):
        wav (torch.Tensor): Raw audio waveform, mono or stereo
        sample_rate (int): Audio sampling rate in Hz
        
    Output:
        torch.Tensor: Speaker embedding of shape [1, 256] (always adds batch dimension)
        
    Features:
        - Automatic audio resampling to 16kHz
        - Stereo to mono conversion if needed
        - Efficient caching of resamplers for different sample rates
        - Device-aware tensor operations
        - Pre-trained on speaker verification tasks
        
    Usage:
        >>> model = SpeakerEmbedding()
        >>> wav, sr = torchaudio.load("speaker_audio.wav")
        >>> embedding = model(wav, sr)  # Shape: [1, 256]
        
    Note:
        Model is set to eval mode and requires_grad=False for inference efficiency.
        Used as a component in SpeakerEmbeddingLDA for complete embedding pipeline.
    """
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        self.device = torch.device(device)

        # Load model directly to target device
        self.model = ResNet293_based()
        state_dict = torch.load(
            ckpt_path,
            weights_only=True,
            mmap=True,
            map_location=self.device  # Direct load to target device
        )
        self.model.load_state_dict(state_dict)
        self.model.featCal = logFbankCal().to(self.device)  # Move fbankCal to device

        # Ensure entire model is on target device
        self.model = self.model.to(self.device)
        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, 16_000).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Prepare audio input for speaker embedding extraction.
        
        Handles common audio preprocessing steps including channel reduction
        and sample rate conversion to the model's expected format.
        
        Args:
            wav (torch.Tensor): Input audio waveform of shape [channels, samples] or [samples]
            sample_rate (int): Original audio sampling rate in Hz
            
        Returns:
            torch.Tensor: Preprocessed audio waveform at 16kHz, mono, shape [1, samples]
            
        Processing Steps:
            1. Convert stereo to mono by averaging channels if needed
            2. Resample to 16kHz using cached resampler
            3. Ensure single channel dimension for model input
            
        Note:
            Resampler instances are cached per sample rate to avoid repeated initialization.
        """
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = self._get_resampler(sample_rate)(wav)
        return wav

    def forward(self, wav: torch.Tensor, sample_rate: int):
        # Ensure wav is already on correct device to avoid transfers
        if wav.device != self.device:
            wav = wav.to(self.device)

        wav = self.prepare_input(wav, sample_rate)
        # Ensure correct dtype but avoid unnecessary transfers
        if wav.dtype != self.dtype:
            wav = wav.to(self.dtype)

        out = self.model(wav)

        # Return tensor on the same device as input to minimize device transfers downstream
        return out  # Already on self.device


class SpeakerEmbeddingLDA(nn.Module):
    """
    Complete speaker embedding system with LDA projection.
    
    Combines ResNet293-based speaker embedding extraction with Linear Discriminant Analysis (LDA)
    projection for improved speaker discrimination. This is the primary interface used by the
    Zonos TTS system for speaker conditioning.
    
    Args:
        device (torch.device, optional): Target compute device. Defaults to DEFAULT_DEVICE.
        
    Input (via forward):
        wav (torch.Tensor): Raw audio waveform, mono or stereo  
        sample_rate (int): Audio sampling rate in Hz
        
    Output:
        Tuple[torch.Tensor, torch.Tensor]: 
            - emb: Original speaker embedding of shape [1, 256]
            - lda_emb: LDA-projected embedding of shape [1, 128]
            
    Model Components:
        1. SpeakerEmbedding: ResNet293-based feature extractor (256-dim output)
        2. LDA projection: Linear layer that projects to 128 dimensions with learned
           discriminant analysis weights trained on speaker verification tasks
           
    Pre-trained Models:
        - Speaker embedding: "ResNet293_SimAM_ASP_base.pt" from Hugging Face Hub
        - LDA projection: "ResNet293_SimAM_ASP_base_LDA-128.pt" from Hugging Face Hub
        - Repository: "Zyphra/Zonos-v0.1-speaker-embedding"
        
    Integration Points:
        - audio_utils.py: make_speaker_embedding() uses LDA embedding for TTS conditioning
        - test_zonos.py: process_speaker_audio() generates speaker embeddings for synthesis
        - model.py: LDA embeddings flow through prefix_conditioning in generate() method
        
    Usage:
        >>> model = SpeakerEmbeddingLDA()
        >>> wav, sr = torchaudio.load("speaker_reference.wav")
        >>> orig_emb, lda_emb = model(wav, sr)
        >>> # Use lda_emb for TTS speaker conditioning
        
    Note:
        The LDA projection improves speaker discrimination by learning optimal linear
        transformations that maximize between-class variance while minimizing within-class variance.
        The 128-dimensional LDA embedding is used for TTS conditioning in the Zonos system.
    """
    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        super().__init__()
        spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base.pt",
            local_dir_use_symlinks = False
        )
        lda_spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base_LDA-128.pt",
            local_dir_use_symlinks = False
        )

        self.device = torch.device(device)

        # Load both models directly to target device
        self.model = SpeakerEmbedding(spk_model_path, device)
        lda_sd = torch.load(lda_spk_model_path, weights_only=True, map_location=self.device)
        out_features, in_features = lda_sd["weight"].shape
        self.lda = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32).to(self.device)
        self.lda.load_state_dict(lda_sd)

        self.requires_grad_(False).eval()

    def forward(self, wav: torch.Tensor, sample_rate: int):
        # Ensure input is on correct device once
        if wav.device != self.device:
            wav = wav.to(self.device)

        emb = self.model(wav, sample_rate)
        # Ensure correct dtype for LDA, transfer if needed
        if emb.dtype != torch.float32:
            emb = emb.to(torch.float32)

        lda_emb = self.lda(emb)

        # Return on same device
        return emb, lda_emb

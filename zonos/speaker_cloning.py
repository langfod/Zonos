import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

from zonos.utils import DEFAULT_DEVICE


class logFbankCal(nn.Module):
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

    def forward(self, x):
        out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        return out


class ASP(nn.Module):
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
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class BasicBlock(nn.Module):
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
    return ResNet(in_planes, SimAMBasicBlock, [10, 20, 64, 3], **kwargs)


class ResNet293_based(nn.Module):
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
    def __init__(self, ckpt_path: str = "ResNet293_SimAM_ASP_base.pt", device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        with torch.device(device):
            self.model = ResNet293_based()
            state_dict = torch.load(ckpt_path, weights_only=True, mmap=True, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.featCal = logFbankCal()

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, 16_000).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = self._get_resampler(sample_rate)(wav)
        return wav

    def forward(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate).to(self.device, self.dtype)
        return self.model(wav).to(wav.device)


class SpeakerEmbeddingLDA(nn.Module):
    def __init__(self, device: str = DEFAULT_DEVICE, enable_cache: bool = True):
        super().__init__()
        spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base.pt",
            local_dir_use_symlinks = False,
        )
        lda_spk_model_path = hf_hub_download(
            repo_id="Zyphra/Zonos-v0.1-speaker-embedding",
            filename="ResNet293_SimAM_ASP_base_LDA-128.pt",
            local_dir_use_symlinks = False,
        )

        self.device = device
        self.enable_cache = enable_cache

        # Cache the model signature to avoid repeated computation
        self._model_signature = "SpeakerEmbeddingLDA_v1"

        # Initialize cache if enabled - use existing global cache
        if self.enable_cache:
            try:
                from utilities.speaker_cache import get_global_speaker_cache
                self._cache = get_global_speaker_cache()
            except ImportError:
                import logging
                logging.warning("Speaker cache not available, disabling caching")
                self.enable_cache = False
                self._cache = None
        else:
            self._cache = None

        with torch.device(device):
            self.model = SpeakerEmbedding(spk_model_path, device)
            lda_sd = torch.load(lda_spk_model_path, weights_only=True)
            out_features, in_features = lda_sd["weight"].shape
            self.lda = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
            self.lda.load_state_dict(lda_sd)

        self.requires_grad_(False).eval()

    def _get_model_signature(self) -> str:
        """Get a signature for the current model configuration."""
        return self._model_signature

    def forward(self, wav: torch.Tensor, sample_rate: int, audio_path: str = None, audio_uuid = None, model_choice: str = None):
        """
        Forward pass with optional caching support.

        Args:
            wav: Audio tensor
            sample_rate: Sample rate of the audio
            audio_path: Optional path to audio file for caching (recommended)
            audio_uuid: Optional UUID for unique identification (C++ unsigned int)
            model_choice: Full model choice string (e.g., "Zyphra/Zonos-v0.1-hybrid")

        Returns:
            Tuple of (base_embedding, lda_embedding)
        """
        # Determine cache model type once
        cache_model_type = model_choice if model_choice else self._model_signature

        # Try to get from cache first if audio_path is provided and caching is enabled
        if self.enable_cache and self._cache and audio_path:
            # Try to get cached result with UUID, specifying target device to avoid extra copies
            cached_result = self._cache.get(audio_path, cache_model_type, audio_uuid, target_device=self.device)
            if cached_result is not None:
                # Cached result should be a tuple of (base_emb, lda_emb)
                if isinstance(cached_result, tuple) and len(cached_result) == 2:
                    base_emb, lda_emb = cached_result
                    # Ensure correct dtype (device should already be correct from cache.get())
                    if base_emb.dtype != wav.dtype:
                        base_emb = base_emb.to(dtype=wav.dtype)
                        lda_emb = lda_emb.to(dtype=wav.dtype)
                    return base_emb, lda_emb

        # Compute embeddings - reduce device transfers
        # Get input tensor on correct device and dtype in one operation
        wav_input = wav.to(self.device, dtype=wav.dtype) if wav.device != self.device or wav.dtype != wav.dtype else wav

        # Compute base embedding
        emb = self.model(wav_input, sample_rate)

        # Convert to float32 for LDA only if needed
        if emb.dtype != torch.float32:
            emb_f32 = emb.to(torch.float32)
        else:
            emb_f32 = emb

        # Compute LDA embedding
        lda_emb = self.lda(emb_f32)

        # Cache the result if enabled and audio_path is provided
        # Only create CPU copies if we're actually going to cache
        if self.enable_cache and self._cache and audio_path:
            try:
                # Efficiently create CPU copies only when needed
                # Detach first to avoid gradients, then move to CPU in one operation
                base_emb_cpu = emb.detach().cpu()
                lda_emb_cpu = lda_emb.detach().cpu()
                cache_data = (base_emb_cpu, lda_emb_cpu)
                self._cache.put(audio_path, cache_data, cache_model_type, audio_uuid)
            except Exception as e:
                import logging
                logging.warning(f"Failed to cache speaker embedding: {e}")

        # Return tensors with consistent dtype (convert LDA back to input dtype if needed)
        if lda_emb.dtype != wav.dtype:
            lda_emb = lda_emb.to(dtype=wav.dtype)
        if emb.dtype != wav.dtype:
            emb = emb.to(dtype=wav.dtype)

        return emb, lda_emb

    def get_cache_stats(self):
        """Get cache statistics if caching is enabled."""
        if self.enable_cache and self._cache:
            return self._cache.get_cache_stats()
        return {"cache_enabled": False}

    def clear_cache(self):
        """Clear the speaker embedding cache."""
        if self.enable_cache and self._cache:
            self._cache.clear_all_cache()

    def print_cache_stats(self):
        """Print cache statistics."""
        if self.enable_cache and self._cache:
            self._cache.print_cache_stats()
        else:
            print("Speaker embedding cache is disabled")

"""Model builders for the challenge.

Track limits: medium <= 10M parameters, small <= 1M parameters.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

MEDIUM_TRACK_LIMIT = 10_000_000
SMALL_TRACK_LIMIT = 1_000_000

# Pretrained RGB weights are ordered R, G, B, which are these Sentinel-2 bands.
RGB_BANDS = ["B04", "B03", "B02"]


def _adapt_first_conv(conv: nn.Conv2d, bands: list[str], pretrained: bool) -> nn.Conv2d:
    """Return a copy of ``conv`` that accepts one input channel per band.

    With pretrained weights, the RGB filters are reused for B04/B03/B02 and every other
    band starts from the mean of the RGB filters. All filters are rescaled so the layer's
    output keeps the size it had with 3 input channels.
    """
    new = nn.Conv2d(
        len(bands), conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            rgb = conv.weight  # (out, 3, k, k)
            mean = rgb.mean(dim=1)
            for i, band in enumerate(bands):
                new.weight[:, i] = rgb[:, RGB_BANDS.index(band)] if band in RGB_BANDS else mean
            new.weight.mul_(3 / len(bands))
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
    return new


def build_efficientnet_b0(bands: list[str], num_classes: int = 10, pretrained: bool = True,
                          dropout: float = 0.2) -> nn.Module:
    """EfficientNet-B0 (about 4M parameters) for multispectral input. Medium track."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    net.features[0][0] = _adapt_first_conv(net.features[0][0], bands, pretrained)
    net.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, num_classes)
    )
    return net


class ShapeFocusedNet(nn.Module):
    """CNN that sees each image standardised by its own statistics, plus a small side branch.

    - Main path: every band of every image is shifted/scaled to mean 0 and spread 1 *within the
      image*, so global brightness, offset and contrast differences between domains disappear
      before the CNN; it can only use patterns (shapes, texture, relative band differences).
    - Statistics branch: the removed per-band means and log-spreads (2 x bands values) go through a
      small MLP. During training this branch is switched off for ``stats_dropout`` of the images, so
      the model cannot rely on it alone (it is needed for flat images such as water).
    - Optional topology branch (``n_topo`` > 0): precomputed per-image features (src/topology.py),
      standardised with stored training statistics, through a small MLP.
    The branch outputs are concatenated with the pooled CNN features before the classifier. The side
    branches have no BatchNorm, so AdaBN only changes the CNN.
    """

    def __init__(self, features: nn.Module, n_features: int, n_bands: int, num_classes: int = 10,
                 dropout: float = 0.2, stats_dropout: float = 0.3, n_topo: int = 0, min_std: float = 0.05):
        super().__init__()
        self.features, self.pool = features, nn.AdaptiveAvgPool2d(1)
        self.stats_dropout, self.min_std, self.n_topo = stats_dropout, min_std, n_topo
        self.stats_mlp = nn.Sequential(nn.Linear(2 * n_bands, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        width = n_features + 64
        if n_topo:
            self.register_buffer("topo_mean", torch.zeros(n_topo))
            self.register_buffer("topo_std", torch.ones(n_topo))
            self.topo_mlp = nn.Sequential(nn.Linear(n_topo, 128), nn.ReLU(), nn.Dropout(0.2),
                                          nn.Linear(128, 64), nn.ReLU())
            width += 64
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, num_classes))

    def set_topo_stats(self, train_features: torch.Tensor) -> None:
        """Store mean/std of the training topology features (saved with the weights)."""
        self.topo_mean.copy_(train_features.float().mean(0))
        self.topo_std.copy_(train_features.float().std(0).clamp_min(1e-6))

    def forward(self, x, extra=None):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        shape = (x - mean) / std.clamp_min(self.min_std)
        stats = torch.cat([mean.flatten(1), torch.log(std.flatten(1) + 1e-3)], dim=1)
        if self.training and self.stats_dropout > 0:
            keep = (torch.rand(len(x), 1, device=x.device) >= self.stats_dropout).float()
            stats = stats * keep
        parts = [self.pool(self.features(shape)).flatten(1), self.stats_mlp(stats)]
        if self.n_topo:
            if extra is None:
                raise ValueError("this model needs the topological features as `extra`")
            parts.append(self.topo_mlp((extra.float() - self.topo_mean) / self.topo_std))
        return self.classifier(torch.cat(parts, dim=1))


def build_efficientnet_b0_shape(bands: list[str], num_classes: int = 10, pretrained: bool = True,
                                n_topo: int = 0, stats_dropout: float = 0.3) -> nn.Module:
    """EfficientNet-B0 features inside ShapeFocusedNet (per-image standardised input)."""
    net = build_efficientnet_b0(bands, num_classes=num_classes, pretrained=pretrained)
    return ShapeFocusedNet(net.features, n_features=net.classifier[1].in_features, n_bands=len(bands),
                           num_classes=num_classes, stats_dropout=stats_dropout, n_topo=n_topo)


def build_mobilenet_v3_small(bands: list[str], num_classes: int = 10, pretrained: bool = True,
                             dropout: float = 0.2) -> nn.Module:
    """MobileNetV3-Small (about 0.93M parameters) for multispectral input. Small track.

    The standard head (576 -> 1024 -> classes) alone has about 0.6M parameters, so it is replaced by
    one linear layer on the pooled features, as in ``build_efficientnet_b0``.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    net.features[0][0] = _adapt_first_conv(net.features[0][0], bands, pretrained)
    net.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(net.classifier[0].in_features, num_classes)
    )
    return net


class TwoBranchNet(nn.Module):
    """One network with two feature branches: EfficientNet-B0 and MobileNetV3-Small.

    - One input layer: a single 3x3 stride-2 convolution (bands -> 32 + 16 channels). Its first 32
      output channels feed the EfficientNet-B0 blocks, the last 16 the MobileNetV3-Small blocks
      (each with its own BatchNorm and activation, as in the original stems).
    - The pooled features of both branches (1280 + 576) are concatenated and classified by one
      linear layer.
    ``init_from`` copies the weights of a trained EfficientNet-B0 and MobileNetV3-Small into it, so
    that before any training it predicts the average of the two models' logits.
    """

    def __init__(self, bands: list[str], num_classes: int = 10, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        big = build_efficientnet_b0(bands, num_classes, pretrained)
        small = build_mobilenet_v3_small(bands, num_classes, pretrained)
        big_conv, small_conv = big.features[0][0], small.features[0][0]
        self.n_big = big_conv.out_channels
        self.stem = nn.Conv2d(len(bands), self.n_big + small_conv.out_channels, kernel_size=3, stride=2,
                              padding=1, bias=False)
        with torch.no_grad():
            self.stem.weight.copy_(torch.cat([big_conv.weight, small_conv.weight]))
        # BatchNorm + activation of each original stem (Conv2dNormActivation cannot be sliced)
        self.big_norm = nn.Sequential(*list(big.features[0])[1:])
        self.small_norm = nn.Sequential(*list(small.features[0])[1:])
        self.big, self.small = big.features[1:], small.features[1:]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(big.classifier[1].in_features + small.classifier[1].in_features, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        a = self.big(self.big_norm(x[:, :self.n_big]))
        b = self.small(self.small_norm(x[:, self.n_big:]))
        return self.classifier(torch.cat([self.pool(a).flatten(1), self.pool(b).flatten(1)], 1))

    @torch.no_grad()
    def init_from(self, big: nn.Module, small: nn.Module) -> None:
        """Copy the weights of a trained EfficientNet-B0 and MobileNetV3-Small (same bands)."""
        self.stem.weight.copy_(torch.cat([big.features[0][0].weight, small.features[0][0].weight]))
        self.big_norm.load_state_dict(nn.Sequential(*list(big.features[0])[1:]).state_dict())
        self.small_norm.load_state_dict(nn.Sequential(*list(small.features[0])[1:]).state_dict())
        self.big.load_state_dict(big.features[1:].state_dict())
        self.small.load_state_dict(small.features[1:].state_dict())
        head = self.classifier[1]
        head.weight.copy_(torch.cat([big.classifier[1].weight, small.classifier[1].weight], 1) / 2)
        head.bias.copy_((big.classifier[1].bias + small.classifier[1].bias) / 2)


MODEL_BUILDERS = {
    "efficientnet_b0": build_efficientnet_b0,
    "efficientnet_b0_shape": build_efficientnet_b0_shape,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "two_branch_effb0_mnv3s": TwoBranchNet,
}


class WithInputLayer(nn.Module):
    """A fixed preprocessing layer (e.g. LowFreqEqualizer) in front of a network."""

    def __init__(self, pre: nn.Module, net: nn.Module):
        super().__init__()
        self.pre, self.net = pre, net

    def forward(self, x):
        return self.net(self.pre(x))


def build_model(name: str, bands: list[str], num_classes: int = 10, input_layer: dict | None = None,
                **kwargs) -> nn.Module:
    """Build a model by name, see MODEL_BUILDERS.

    ``input_layer`` adds a fixed first layer, e.g.
    ``{"type": "low_freq_equalizer", "template": [...], "radius": 0.05}``.
    """
    if input_layer is not None and input_layer["type"] == "spectral_indices":
        # the network sees the bands plus one extra channel per index (src/indices.py)
        from src.indices import SpectralIndexLayer
        params = {k: v for k, v in input_layer.items() if k not in ("type", "bands")}
        pre = SpectralIndexLayer(bands=bands, **params)
        net = MODEL_BUILDERS[name](bands=bands + pre.indices, num_classes=num_classes, **kwargs)
        return WithInputLayer(pre, net)
    net = MODEL_BUILDERS[name](bands=bands, num_classes=num_classes, **kwargs)
    if input_layer is None:
        return net
    if input_layer["type"] == "low_freq_equalizer":
        from src.fourier import LowFreqEqualizer
        params = {k: v for k, v in input_layer.items() if k != "type"}
        return WithInputLayer(LowFreqEqualizer(**params), net)
    raise ValueError(f"unknown input layer {input_layer['type']}")

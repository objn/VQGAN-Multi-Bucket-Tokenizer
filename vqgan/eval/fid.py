import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_feature_extractor = None


class InceptionFeatures(nn.Module):
    """Pretrained InceptionV3 pool features (2048-d), used only as an
    evaluation-time metric (FID) — this doesn't affect how the generative
    model itself is trained, same as the existing pretrained-VGG LPIPS loss.
    """

    def __init__(self):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

    def forward(self, x):
        # x: [B, 3, H, W] in [-1, 1]
        x = (x + 1) / 2
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.net(x)


def get_feature_extractor(device):
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = InceptionFeatures().to(device)
    return _feature_extractor


@torch.no_grad()
def extract_features(model: InceptionFeatures, images: torch.Tensor) -> np.ndarray:
    return model(images).cpu().numpy()


def compute_statistics(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def fid_from_stats(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

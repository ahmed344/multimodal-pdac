"""Adversarial latent fusion for sparse MALDI-MSI and IHC densities."""

from .losses import ZILNLoss
from .model import AdversarialLatentFusion

__all__ = ["AdversarialLatentFusion", "ZILNLoss"]

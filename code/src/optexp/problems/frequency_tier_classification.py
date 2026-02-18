"""
Classification problem with grouped metrics by frequency tiers.

For synthetic datasets where classes are organized into frequency tiers
(e.g., tier 0: 64 samples/class, tier 1: 32 samples/class, etc.)
"""
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from typing import List

from optexp.problems.classification import Classification
from optexp.problems.fb_problem import FullBatchProblem


class AccuracyByFrequencyTier(nn.Module):
    """Compute accuracy separately for each frequency tier.
    
    For synthetic dataset with size=m, there are m frequency tiers:
    - Tier 0: 2^(m-1) samples per class
    - Tier 1: 2^(m-2) samples per class
    - ...
    - Tier m-1: 1 sample per class
    
    Args:
        tier_boundaries: List of class indices marking tier boundaries
            e.g., [1, 3, 7, 15, 31, 63, 127] for size=7
    """
    def __init__(self, tier_boundaries: List[int]) -> None:
        super().__init__()
        self.tier_boundaries = tier_boundaries
        self.num_tiers = len(tier_boundaries)
        self.__name__ = "AccuracyByFrequencyTier"

    def __str__(self):
        return f"AccuracyByFrequencyTier(tiers={self.num_tiers})"

    def forward(self, inputs, labels):
        """Returns tuple of (values, counts) for each tier's accuracy."""
        classes = torch.argmax(inputs, dim=1)
        correct = (classes == labels).float()
        
        values = []
        counts = []
        
        # Tier 0 (most frequent)
        tier_mask = labels < self.tier_boundaries[0]
        tier_correct = correct[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
        tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
        values.append(tier_correct)
        counts.append(tier_count)
        
        # Middle tiers
        for i in range(len(self.tier_boundaries) - 1):
            tier_mask = (labels >= self.tier_boundaries[i]) & (labels < self.tier_boundaries[i+1])
            tier_correct = correct[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
            tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
            values.append(tier_correct)
            counts.append(tier_count)
        
        # Last tier (least frequent)
        tier_mask = labels >= self.tier_boundaries[-1]
        tier_correct = correct[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
        tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
        values.append(tier_correct)
        counts.append(tier_count)
        
        # Stack into tensors
        values_tensor = torch.stack(values)
        counts_tensor = torch.stack(counts)
        
        return values_tensor, counts_tensor


class CrossEntropyLossByFrequencyTier(nn.Module):
    """Compute cross-entropy loss separately for each frequency tier.
    
    Args:
        tier_boundaries: List of class indices marking tier boundaries
    """
    def __init__(self, tier_boundaries: List[int]) -> None:
        super().__init__()
        self.tier_boundaries = tier_boundaries
        self.num_tiers = len(tier_boundaries)
        self.__name__ = "CrossEntropyLossByFrequencyTier"

    def __str__(self):
        return f"CrossEntropyLossByFrequencyTier(tiers={self.num_tiers})"

    def forward(self, inputs, labels):
        """Returns tuple of (values, counts) for each tier's loss."""
        losses = cross_entropy(inputs, labels, reduction="none")
        
        values = []
        counts = []
        
        # Tier 0 (most frequent)
        tier_mask = labels < self.tier_boundaries[0]
        tier_loss = losses[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
        tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
        values.append(tier_loss)
        counts.append(tier_count)
        
        # Middle tiers
        for i in range(len(self.tier_boundaries) - 1):
            tier_mask = (labels >= self.tier_boundaries[i]) & (labels < self.tier_boundaries[i+1])
            tier_loss = losses[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
            tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
            values.append(tier_loss)
            counts.append(tier_count)
        
        # Last tier (least frequent)
        tier_mask = labels >= self.tier_boundaries[-1]
        tier_loss = losses[tier_mask].sum() if tier_mask.any() else torch.tensor(0.0, device=inputs.device)
        tier_count = tier_mask.sum().float() if tier_mask.any() else torch.tensor(1.0, device=inputs.device)
        values.append(tier_loss)
        counts.append(tier_count)
        
        # Stack into tensors
        values_tensor = torch.stack(values)
        counts_tensor = torch.stack(counts)
        
        return values_tensor, counts_tensor


class ClassificationWithFrequencyTierStats(Classification):
    """Classification problem that logs grouped metrics by frequency tier.
    
    Logs standard metrics (overall loss, accuracy) plus tier-specific metrics.
    
    Args:
        model: Neural network model
        dataset: Dataset object
        tier_boundaries: List of class indices marking tier boundaries
            For size=7: [1, 3, 7, 15, 31, 63, 127]
            Tier 0: classes 0-0 (1 class, 64 samples each)
            Tier 1: classes 1-2 (2 classes, 32 samples each)
            Tier 2: classes 3-6 (4 classes, 16 samples each)
            Tier 3: classes 7-14 (8 classes, 8 samples each)
            Tier 4: classes 15-30 (16 classes, 4 samples each)
            Tier 5: classes 31-62 (32 classes, 2 samples each)
            Tier 6: classes 63-126 (64 classes, 1 sample each)
    """
    def __init__(self, model, dataset, tier_boundaries: List[int]):
        super().__init__(model, dataset)
        self.tier_boundaries = tier_boundaries
        self.num_tiers = len(tier_boundaries)

    def get_criterions(self) -> List[nn.Module]:
        """Return list of criteria including tier-specific metrics."""
        criterions = super().get_criterions()
        criterions.append(AccuracyByFrequencyTier(self.tier_boundaries))
        criterions.append(CrossEntropyLossByFrequencyTier(self.tier_boundaries))
        return criterions

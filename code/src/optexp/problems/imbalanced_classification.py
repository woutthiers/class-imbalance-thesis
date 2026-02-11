"""
Classification problem with grouped metrics for majority and minority classes.

Useful for imbalanced datasets where you want to track performance on:
- Majority classes (common, well-represented classes)
- Minority classes (rare, underrepresented classes)

Without logging metrics for every individual class.
"""
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from typing import List

from optexp.problems.classification import Classification
from optexp.problems.fb_problem import FullBatchProblem
from optexp.problems.utils import Accuracy


class AccuracyMajorityMinority(nn.Module):
    """Compute accuracy separately for majority and minority classes.
    
    Returns a tuple of tensors for compatibility with the framework.
    
    Args:
        num_majority_classes: Number of majority/common classes (e.g., 10)
    """
    def __init__(self, num_majority_classes: int = 10) -> None:
        super().__init__()
        self.num_majority_classes = num_majority_classes
        self.__name__ = "AccuracyMajorityMinority"

    def __str__(self):
        return "AccuracyMajorityMinority()"

    def forward(self, inputs, labels):
        """Returns tuple of (values, counts) for majority and minority accuracy."""
        classes = torch.argmax(inputs, dim=1)
        correct = (classes == labels).float()
        
        # Separate majority and minority samples
        majority_mask = labels < self.num_majority_classes
        minority_mask = labels >= self.num_majority_classes
        
        # Compute accuracies and counts
        majority_correct = correct[majority_mask].sum() if majority_mask.any() else torch.tensor(0.0, device=inputs.device)
        majority_count = majority_mask.sum().float() if majority_mask.any() else torch.tensor(1.0, device=inputs.device)
        
        minority_correct = correct[minority_mask].sum() if minority_mask.any() else torch.tensor(0.0, device=inputs.device)
        minority_count = minority_mask.sum().float() if minority_mask.any() else torch.tensor(1.0, device=inputs.device)
        
        # Stack into tensors: [majority_acc, minority_acc]
        values = torch.stack([majority_correct, minority_correct])
        counts = torch.stack([majority_count, minority_count])
        
        return values, counts


class CrossEntropyLossMajorityMinority(nn.Module):
    """Compute cross-entropy loss separately for majority and minority classes.
    
    Returns a tuple of tensors for compatibility with the framework.
    
    Args:
        num_majority_classes: Number of majority/common classes (e.g., 10)
    """
    def __init__(self, num_majority_classes: int = 10) -> None:
        super().__init__()
        self.num_majority_classes = num_majority_classes
        self.__name__ = "CrossEntropyLossMajorityMinority"

    def __str__(self):
        return "CrossEntropyLossMajorityMinority()"

    def forward(self, inputs, labels):
        """Returns tuple of (values, counts) for majority and minority loss."""
        losses = cross_entropy(inputs, labels, reduction="none")
        
        # Separate majority and minority samples
        majority_mask = labels < self.num_majority_classes
        minority_mask = labels >= self.num_majority_classes
        
        # Compute losses and counts
        majority_loss = losses[majority_mask].sum() if majority_mask.any() else torch.tensor(0.0, device=inputs.device)
        majority_count = majority_mask.sum().float() if majority_mask.any() else torch.tensor(1.0, device=inputs.device)
        
        minority_loss = losses[minority_mask].sum() if minority_mask.any() else torch.tensor(0.0, device=inputs.device)
        minority_count = minority_mask.sum().float() if minority_mask.any() else torch.tensor(1.0, device=inputs.device)
        
        # Stack into tensors: [majority_loss, minority_loss]
        values = torch.stack([majority_loss, minority_loss])
        counts = torch.stack([majority_count, minority_count])
        
        return values, counts


class ClassificationWithMajorityMinorityStats(Classification):
    """Classification problem that logs grouped metrics for majority/minority classes.
    
    Logs standard metrics (overall loss, accuracy) plus:
    - majority_acc / minority_acc
    - majority_loss / minority_loss
    
    Args:
        num_majority_classes: How many classes are considered "majority" (default: 10)
    """
    def __init__(self, model, dataset, num_majority_classes: int = 10):
        self.num_majority_classes = num_majority_classes
        super().__init__(model, dataset)
    
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
            CrossEntropyLossMajorityMinority(self.num_majority_classes),
            AccuracyMajorityMinority(self.num_majority_classes),
        ]


class FullBatchClassificationWithMajorityMinorityStats(
    ClassificationWithMajorityMinorityStats, FullBatchProblem
):
    """Full-batch version of ClassificationWithMajorityMinorityStats."""
    pass

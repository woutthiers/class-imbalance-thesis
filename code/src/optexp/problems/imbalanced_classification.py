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
    
    Args:
        num_majority_classes: Number of majority/common classes (e.g., 10)
    """
    def __init__(self, num_majority_classes: int = 10) -> None:
        super().__init__()
        self.num_majority_classes = num_majority_classes

    def forward(self, inputs, labels):
        """Returns dict with 'majority_acc' and 'minority_acc'."""
        classes = torch.argmax(inputs, dim=1)
        correct = (classes == labels).float()
        
        # Separate majority and minority samples
        majority_mask = labels < self.num_majority_classes
        minority_mask = labels >= self.num_majority_classes
        
        result = {}
        
        # Majority accuracy
        if majority_mask.any():
            result['majority_acc'] = correct[majority_mask].mean().item()
        else:
            result['majority_acc'] = 0.0
        
        # Minority accuracy
        if minority_mask.any():
            result['minority_acc'] = correct[minority_mask].mean().item()
        else:
            result['minority_acc'] = 0.0
        
        return result


class CrossEntropyLossMajorityMinority(nn.Module):
    """Compute cross-entropy loss separately for majority and minority classes.
    
    Args:
        num_majority_classes: Number of majority/common classes (e.g., 10)
    """
    def __init__(self, num_majority_classes: int = 10) -> None:
        super().__init__()
        self.num_majority_classes = num_majority_classes

    def forward(self, inputs, labels):
        """Returns dict with 'majority_loss' and 'minority_loss'."""
        losses = cross_entropy(inputs, labels, reduction="none")
        
        # Separate majority and minority samples
        majority_mask = labels < self.num_majority_classes
        minority_mask = labels >= self.num_majority_classes
        
        result = {}
        
        # Majority loss
        if majority_mask.any():
            result['majority_loss'] = losses[majority_mask].mean().item()
        else:
            result['majority_loss'] = 0.0
        
        # Minority loss
        if minority_mask.any():
            result['minority_loss'] = losses[minority_mask].mean().item()
        else:
            result['minority_loss'] = 0.0
        
        return result


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

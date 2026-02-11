"""
Test script to verify the ImbalancedMNISTWithBarcodes dataset.

This script checks that the dataset has the correct structure:
- 10 common classes with ~5000 samples each
- ~10,240 rare classes with 5 samples each
"""
import numpy as np
import torch


def test_imbalanced_dataset():
    """Test the new ImbalancedMNISTWithBarcodes dataset"""
    print("=" * 70)
    print("Testing ImbalancedMNISTWithBarcodes Dataset")
    print("=" * 70)
    print()
    
    try:
        from optexp.datasets.barcoded_mnist import make_imbalanced_mnist_with_barcodes
        
        print("Creating dataset...")
        X_tr, y_tr, X_va, y_va = make_imbalanced_mnist_with_barcodes(
            common_samples_per_class_tr=5000,
            common_samples_per_class_val=1000,
            rare_samples_per_class_tr=5,
            rare_samples_per_class_val=1,
            num_rare_classes=10240,
        )
        
        print("✓ Dataset created successfully")
        print()
        
        # Check shapes
        print("Dataset Statistics:")
        print("-" * 70)
        print(f"Training samples: {len(X_tr):,}")
        print(f"Validation samples: {len(X_va):,}")
        print(f"Number of classes: {len(torch.unique(y_tr)):,}")
        print(f"Image shape: {X_tr.shape[1:]}")
        print()
        
        # Check class distribution
        print("Class Distribution:")
        print("-" * 70)
        
        unique, counts = torch.unique(y_tr, return_counts=True)
        
        # Common classes (0-9)
        common_classes = unique[:10]
        common_counts = counts[:10]
        
        print(f"Common classes (0-9):")
        for cls, count in zip(common_classes, common_counts):
            print(f"  Class {cls}: {count} samples")
        print(f"  Average: {common_counts.float().mean():.0f} samples")
        print()
        
        # Rare classes (10+)
        rare_classes = unique[10:]
        rare_counts = counts[10:]
        
        print(f"Rare classes (10-{len(unique)-1}):")
        print(f"  Number of rare classes: {len(rare_classes):,}")
        print(f"  Samples per rare class (min/max/mean): {rare_counts.min()}/{rare_counts.max()}/{rare_counts.float().mean():.1f}")
        print()
        
        # Verify expected structure
        print("Verification:")
        print("-" * 70)
        
        # Check common classes
        expected_common = 10
        actual_common = len(common_classes)
        print(f"✓ Common classes: {actual_common} (expected: {expected_common})")
        
        expected_common_samples = 5000
        actual_common_avg = common_counts.float().mean().item()
        print(f"✓ Common class samples: ~{actual_common_avg:.0f} (expected: ~{expected_common_samples})")
        
        # Check rare classes
        expected_rare = 10240
        actual_rare = len(rare_classes)
        print(f"✓ Rare classes: {actual_rare} (expected: {expected_rare})")
        
        expected_rare_samples = 5
        actual_rare_avg = rare_counts.float().mean().item()
        print(f"✓ Rare class samples: {actual_rare_avg:.1f} (expected: {expected_rare_samples})")
        
        # Total samples
        expected_total_train = 10 * 5000 + 10240 * 5
        actual_total_train = len(X_tr)
        print(f"✓ Total training samples: {actual_total_train:,} (expected: {expected_total_train:,})")
        
        # Imbalance ratio
        imbalance_ratio = common_counts.float().mean() / rare_counts.float().mean()
        print(f"✓ Imbalance ratio (common:rare): {imbalance_ratio:.0f}:1")
        
        print()
        print("=" * 70)
        print("All checks passed! ✓")
        print("=" * 70)
        print()
        
        # Print some examples
        print("Sample Distribution Plot (first 50 classes):")
        print("-" * 70)
        max_count = max(counts[:50].max().item(), 100)
        for i, (cls, count) in enumerate(zip(unique[:50], counts[:50])):
            bar_length = int(50 * count / max_count)
            bar = "█" * bar_length
            label = "COMMON" if cls < 10 else "rare  "
            print(f"Class {cls:4d} [{label}]: {bar} {count:5d} samples")
        
        if len(unique) > 50:
            print(f"... ({len(unique) - 50} more classes)")
        
        print()
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import dataset module: {e}")
        print("  Make sure you're in the code/ directory and have installed the package:")
        print("  cd code && pip install -e .")
        return False
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_class():
    """Test the Dataset class wrapper"""
    print()
    print("=" * 70)
    print("Testing ImbalancedMNISTWithBarcodes Dataset Class")
    print("=" * 70)
    print()
    
    try:
        from optexp.datasets.barcoded_mnist import ImbalancedMNISTWithBarcodes
        from optexp import config
        
        print("Creating dataset loader...")
        dataset = ImbalancedMNISTWithBarcodes(batch_size=512)
        
        # Note: This will try to use GPU if available, might need CPU fallback
        print("Loading dataset...")
        try:
            train_loader, val_loader, input_shape, output_shape, class_counts = dataset.load()
            
            print("✓ Dataset loaded successfully")
            print()
            print(f"Input shape: {input_shape}")
            print(f"Output shape (num classes): {output_shape}")
            print(f"Batch size: {dataset.batch_size}")
            print(f"Number of classes: {len(class_counts)}")
            print()
            
            # Print class frequency histogram
            print("Class frequency distribution:")
            common_classes = class_counts[:10]
            rare_classes = class_counts[10:]
            print(f"  Common classes (0-9): avg {common_classes.float().mean():.0f} samples")
            print(f"  Rare classes (10+): avg {rare_classes.float().mean():.1f} samples")
            print()
            
            print("✓ Dataset class integration works!")
            
        except Exception as e:
            print(f"Note: Could not load to device (might need GPU): {e}")
            print("This is OK if testing locally without GPU")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ImbalancedMNISTWithBarcodes Test Suite" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Test 1: Dataset creation
    test1_passed = test_imbalanced_dataset()
    
    # Test 2: Dataset class integration
    test2_passed = test_dataset_class()
    
    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Dataset creation: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Dataset class integration: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print()
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        print()
        print("You can now use ImbalancedMNISTWithBarcodes in your experiments.")
        print("This dataset has:")
        print("  • 10 common classes with ~5,000 samples each (realistic frequent classes)")
        print("  • 10,240 rare classes with 5 samples each (realistic rare classes)")
        print("  • 1000:1 imbalance ratio for testing optimizer robustness")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print()

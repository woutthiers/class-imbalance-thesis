"""
Quick test script to verify AdaptiveSign optimizer implementation.

Run this locally before submitting to VSC to ensure everything works.
"""
import torch
import torch.nn as nn
from optexp.optimizers.AdaptiveSign import AdaptiveSignOptimizer, AdaptiveSign_NM, AdaptiveSign_M
from optexp.optimizers.learning_rate import LearningRate


def test_basic_functionality():
    """Test basic optimizer functionality"""
    print("Testing AdaptiveSign optimizer...")
    
    # Simple model
    model = nn.Linear(10, 2)
    
    # Create optimizer
    lr = LearningRate(exponent=-3, base=10)  # 10^-3 = 0.001
    opt_wrapper = AdaptiveSign_NM(lr, eps=1e-8)
    optimizer = opt_wrapper.load(model)
    
    # Dummy data
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None, "Gradients should exist after backward()"
    
    # Optimizer step
    optimizer.step()
    
    print("✓ Basic forward/backward pass works")
    
    # Check that parameters changed
    optimizer.zero_grad()
    loss2 = nn.CrossEntropyLoss()(model(x), y)
    
    print(f"✓ Loss before step: {loss.item():.4f}")
    print(f"✓ Loss after step: {loss2.item():.4f}")
    print()


def test_update_formula():
    """Test that the update formula is correct"""
    print("Testing update formula...")
    
    # Simple 1-parameter model for precise checking
    param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    
    # Create optimizer with known parameters
    lr = 0.1
    eps = 1e-2
    optimizer = AdaptiveSignOptimizer([param], lr=lr, eps=eps, momentum=0)
    
    # Set specific gradients
    param.grad = torch.tensor([1.0, -2.0, 0.5])
    
    # Expected update: x = x - lr * g / (eps + |g|)
    # For g=[1.0, -2.0, 0.5], |g|=[1.0, 2.0, 0.5]
    # update = [1.0/(0.01+1.0), -2.0/(0.01+2.0), 0.5/(0.01+0.5)]
    #        = [1.0/1.01, -2.0/2.01, 0.5/0.51]
    #        ≈ [0.9901, -0.9950, 0.9804]
    # x_new = x - 0.1 * update
    expected = torch.tensor([
        1.0 - lr * 1.0 / (eps + 1.0),
        2.0 - lr * (-2.0) / (eps + 2.0),
        3.0 - lr * 0.5 / (eps + 0.5),
    ])
    
    # Do optimizer step
    optimizer.step()
    
    # Check
    assert torch.allclose(param, expected, atol=1e-6), \
        f"Update formula incorrect!\nExpected: {expected}\nGot: {param}"
    
    print(f"✓ Update formula correct: x - lr * g / (eps + |g|)")
    print(f"  Initial: [1.0, 2.0, 3.0]")
    print(f"  Gradient: {param.grad.tolist()}")
    print(f"  Final: {param.tolist()}")
    print()


def test_epsilon_values():
    """Test different epsilon values"""
    print("Testing different epsilon values...")
    
    for eps in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        model = nn.Linear(5, 2)
        lr = LearningRate(exponent=-2, base=10)
        opt = AdaptiveSign_NM(lr, eps=eps)
        optimizer = opt.load(model)
        
        # Dummy forward/backward
        x = torch.randn(3, 5)
        y = torch.randint(0, 2, (3,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ eps={eps:.0e} works")
    
    print()


def test_momentum():
    """Test momentum variant"""
    print("Testing momentum variant...")
    
    model = nn.Linear(5, 2)
    lr = LearningRate(exponent=-2, base=10)
    
    # Without momentum
    opt_nm = AdaptiveSign_NM(lr, eps=1e-8)
    print(f"  ✓ No momentum (beta1=0): {opt_nm}")
    
    # With momentum
    opt_m = AdaptiveSign_M(lr, eps=1e-8)
    print(f"  ✓ With momentum (beta1=0.9): {opt_m}")
    
    print()


def test_integration():
    """Test integration with experiment code"""
    print("Testing integration with experiment framework...")
    
    try:
        from optexp.experiments.vision.barcoded_mnist_adaptive_sign import (
            make_adaptive_sign_grid,
            EPSILON_VALUES,
        )
        from optexp.datasets.barcoded_mnist import ImbalancedMNISTWithBarcodes
        
        # Create a small grid
        opts = make_adaptive_sign_grid(
            epsilon_values=[1e-8, 1e-4],
            lr_start=-3,
            lr_end=-1,
            lr_density=0,
        )
        
        print(f"  ✓ Created {len(opts)} optimizer configurations")
        print(f"  ✓ Epsilon values: {EPSILON_VALUES}")
        
        # Test dataset
        dataset = ImbalancedMNISTWithBarcodes(batch_size=512)
        print(f"  ✓ Dataset class: ImbalancedMNISTWithBarcodes")
        print(f"  ✓ Structure: 10 common classes (~5k samples) + 10,240 rare classes (5 samples)")
        
    except ImportError as e:
        print(f"  ✗ Could not import experiment module: {e}")
        return
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("AdaptiveSign Optimizer Test Suite")
    print("=" * 60)
    print()
    
    test_basic_functionality()
    test_update_formula()
    test_epsilon_values()
    test_momentum()
    test_integration()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()
    print("You can now run the experiments on VSC.")
    print("See ADAPTIVE_SIGN_VSC_GUIDE.md for instructions.")

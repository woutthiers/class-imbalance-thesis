"""
Visualization script to understand AdaptiveSign optimizer behavior.

This script creates plots showing how different epsilon values affect
the normalization of gradients.
"""
import numpy as np
import matplotlib.pyplot as plt


def adaptive_sign_transform(g, eps):
    """
    Apply AdaptiveSign transformation: g / (eps + |g|)
    
    Args:
        g: Gradient value
        eps: Epsilon parameter
    
    Returns:
        Transformed gradient
    """
    return g / (eps + np.abs(g))


def plot_normalization_effect():
    """Plot how different epsilon values affect gradient transformation"""
    
    # Range of gradient values
    g = np.linspace(-5, 5, 1000)
    
    # Different epsilon values
    epsilon_values = [1e-8, 1e-4, 1e-2, 1e-1, 1.0]
    
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Transformation curves
    plt.subplot(2, 2, 1)
    for eps in epsilon_values:
        normalized = adaptive_sign_transform(g, eps)
        plt.plot(g, normalized, label=f'ε={eps:.0e}', linewidth=2)
    
    plt.plot(g, g, 'k--', label='SGD (no normalization)', alpha=0.5)
    plt.plot(g, np.sign(g), 'k:', label='SignSGD', alpha=0.5)
    plt.xlabel('Original Gradient (g)', fontsize=12)
    plt.ylabel('Normalized Gradient g/(ε+|g|)', fontsize=12)
    plt.title('AdaptiveSign Normalization for Different ε', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 5)
    plt.ylim(-1.5, 1.5)
    
    # Plot 2: Zoom on small gradients
    plt.subplot(2, 2, 2)
    g_small = np.linspace(-0.1, 0.1, 1000)
    for eps in epsilon_values:
        normalized = adaptive_sign_transform(g_small, eps)
        plt.plot(g_small, normalized, label=f'ε={eps:.0e}', linewidth=2)
    
    plt.plot(g_small, g_small, 'k--', label='SGD', alpha=0.5)
    plt.xlabel('Original Gradient (g)', fontsize=12)
    plt.ylabel('Normalized Gradient', fontsize=12)
    plt.title('Zoom: Effect on Small Gradients', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Amplification factor
    plt.subplot(2, 2, 3)
    g_positive = np.linspace(0.001, 5, 1000)
    for eps in epsilon_values:
        # How much the gradient is amplified/compressed
        factor = 1 / (eps + g_positive)
        plt.semilogy(g_positive, factor, label=f'ε={eps:.0e}', linewidth=2)
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No change')
    plt.xlabel('Original Gradient Magnitude |g|', fontsize=12)
    plt.ylabel('Amplification Factor 1/(ε+|g|)', fontsize=12)
    plt.title('How Much Gradients Are Amplified', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Example distribution
    plt.subplot(2, 2, 4)
    
    # Simulate gradient distribution (mixture of small and large)
    np.random.seed(42)
    gradients = np.concatenate([
        np.random.normal(0, 0.01, 500),  # Many small gradients (rare classes)
        np.random.normal(0, 1.0, 100),   # Some large gradients (common classes)
    ])
    
    for eps in [1e-8, 1e-2, 1.0]:
        normalized = adaptive_sign_transform(gradients, eps)
        plt.hist(normalized, bins=50, alpha=0.5, label=f'ε={eps:.0e}', density=True)
    
    plt.xlabel('Normalized Gradient Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Effect on Gradient Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_sign_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to adaptive_sign_visualization.png")
    plt.show()


def print_examples():
    """Print concrete examples of the transformation"""
    
    print("\n" + "="*70)
    print("AdaptiveSign Transformation Examples")
    print("="*70)
    
    gradient_examples = [0.001, 0.01, 0.1, 1.0, 10.0]
    epsilon_values = [1e-8, 1e-4, 1e-2, 1.0]
    
    print("\nUpdate = - learning_rate × g/(ε + |g|)")
    print("\nAssuming learning_rate = 0.1:")
    print()
    
    # Header
    print(f"{'Gradient g':<12}", end="")
    for eps in epsilon_values:
        print(f"{'ε=' + f'{eps:.0e}':<15}", end="")
    print()
    print("-" * 70)
    
    # Data rows
    for g in gradient_examples:
        print(f"{g:<12.3f}", end="")
        for eps in epsilon_values:
            normalized = adaptive_sign_transform(g, eps)
            update = -0.1 * normalized
            print(f"{update:<15.6f}", end="")
        print()
    
    print("\n" + "="*70)
    print("Key Observations:")
    print("="*70)
    print()
    print("1. SMALL ε (1e-8): Almost like SignSGD")
    print("   → All gradients become ±1, update = ±0.1")
    print("   → Very strong normalization")
    print()
    print("2. MEDIUM ε (1e-4, 1e-2): Balanced normalization")
    print("   → Small gradients amplified (0.001 → ~0.1 update)")
    print("   → Large gradients compressed (10.0 → ~0.1 update)")
    print("   → Sweet spot for imbalanced data")
    print()
    print("3. LARGE ε (1.0): Almost like SGD")
    print("   → Small gradients stay small (0.001 → ~0.0001 update)")
    print("   → Large gradients stay large (10.0 → ~0.9 update)")
    print("   → Minimal normalization")
    print()


def compare_with_adam():
    """Compare AdaptiveSign with Adam's normalization"""
    
    print("\n" + "="*70)
    print("Comparison: AdaptiveSign vs Adam")
    print("="*70)
    print()
    
    print("ADAM:")
    print("  Update = - lr × m / (√v + ε)")
    print("  where:")
    print("    m = exponential moving average of gradients")
    print("    v = exponential moving average of squared gradients")
    print()
    print("ADAPTIVE SIGN:")
    print("  Update = - lr × m / (ε + |m|)")
    print("  where:")
    print("    m = gradient (or momentum buffer)")
    print("    NO second moment (v)")
    print()
    
    print("Key Differences:")
    print("  1. Adam uses √(v) = RMS of past gradients")
    print("  2. AdaptiveSign uses |m| = current gradient magnitude")
    print("  3. Adam adapts slowly (exponential average)")
    print("  4. AdaptiveSign adapts instantly (current gradient)")
    print()
    
    print("Why This Might Help for Imbalanced Data:")
    print("  • No accumulation of squared gradients from frequent classes")
    print("  • Direct normalization treats each gradient independently")
    print("  • Epsilon controls how aggressive the normalization is")
    print("  • Can amplify gradients from rare classes more effectively")
    print()


if __name__ == "__main__":
    print("="*70)
    print("AdaptiveSign Optimizer Visualization")
    print("="*70)
    
    # Print text examples
    print_examples()
    compare_with_adam()
    
    # Create plots
    print("\n" + "="*70)
    print("Creating visualization plots...")
    print("="*70)
    print()
    
    try:
        plot_normalization_effect()
    except Exception as e:
        print(f"Could not create plots (matplotlib might not be available): {e}")
        print("But the examples above show the key behavior!")
    
    print()
    print("="*70)
    print("Done! Review the examples and plots to understand the optimizer.")
    print("="*70)

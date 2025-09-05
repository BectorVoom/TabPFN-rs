#!/usr/bin/env python3
"""
Python/PyTorch test for MLP canonical values - cross-language verification with Rust/Burn
Implements exact specification from the MLP testing plan.
"""

import torch
import torch.nn.functional as F
import math
import sys

def gelu_exact(t):
    """
    Exact GELU using erf formula to match Rust implementation.
    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return t * 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))

def test_mlp_canonical_fixture():
    """Test MLP with canonical fixture values for cross-language verification"""
    print("🧪 Testing canonical MLP values (Python/PyTorch)")
    
    # Canonical fixtures from specification
    x = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32)  # shape (3,)
    
    # Weight matrices in PyTorch layout [out_features, in_features]
    W1 = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.0, -0.1, 0.2],
        [0.5, 0.5, 0.5],
        [-0.2, 0.1, 0.0]
    ], dtype=torch.float32)  # shape (4,3)
    
    W2 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=torch.float32)  # shape (3,4)
    
    # Expected canonical output from specification
    expected = torch.tensor([0.11585194, -0.13783130, 0.84134475], dtype=torch.float32)
    
    # Manual forward pass
    def mlp_forward(x):
        # Linear1: no bias
        hidden = W1.matmul(x)   # shape (4,)
        
        # GELU activation (exact erf formula)
        activated = gelu_exact(hidden)
        
        # Linear2: no bias  
        output = W2.matmul(activated)
        return output
    
    # Forward pass
    output = mlp_forward(x)
    
    print(f"   Input: {x.tolist()}")
    print(f"   Output: {output.tolist()}")
    print(f"   Expected: {expected.tolist()}")
    
    # Verify each element
    differences = torch.abs(output - expected)
    for i, (actual, expected_val, diff) in enumerate(zip(output, expected, differences)):
        print(f"   Diff[{i}]: {diff.item():.10f} (actual: {actual.item():.8f}, expected: {expected_val.item():.8f})")
    
    # Test with tolerance for float32 precision
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6), f"Canonical values mismatch. Max diff: {differences.max().item()}"
    
    print("✅ Canonical MLP values verified (Python)")
    return output

def test_intermediate_values():
    """Test intermediate computation steps for debugging"""
    print("\n🔍 Verifying intermediate computation steps")
    
    x = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32)
    W1 = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.0, -0.1, 0.2], 
        [0.5, 0.5, 0.5],
        [-0.2, 0.1, 0.0]
    ], dtype=torch.float32)
    
    # Step 1: Linear1 computation
    hidden = W1.matmul(x)
    expected_hidden = torch.tensor([0.2, -0.4, 1.0, 0.0], dtype=torch.float32)
    print(f"   Hidden layer: {hidden.tolist()}")
    print(f"   Expected hidden: {expected_hidden.tolist()}")
    assert torch.allclose(hidden, expected_hidden, atol=1e-7), "Hidden layer computation mismatch"
    
    # Step 2: GELU activation
    activated = gelu_exact(hidden)
    expected_activated = torch.tensor([0.11585194, -0.13783130, 0.84134475, 0.0], dtype=torch.float32)
    print(f"   Activated: {activated.tolist()}")
    print(f"   Expected activated: {expected_activated.tolist()}")
    
    # More lenient tolerance for GELU since it involves erf computation
    assert torch.allclose(activated, expected_activated, atol=1e-6), "GELU activation mismatch"
    
    print("✅ Intermediate values verified")

def test_gelu_precision():
    """Test GELU function precision with known values"""
    print("\n🎯 Testing GELU precision")
    
    test_values = torch.tensor([0.2, -0.4, 1.0, 0.0], dtype=torch.float32)
    expected_gelu = torch.tensor([0.11585194, -0.13783130, 0.84134475, 0.0], dtype=torch.float32)
    
    # Test exact erf-based GELU
    gelu_output = gelu_exact(test_values)
    print(f"   Input: {test_values.tolist()}")
    print(f"   GELU output: {gelu_output.tolist()}")
    print(f"   Expected: {expected_gelu.tolist()}")
    
    differences = torch.abs(gelu_output - expected_gelu)
    for i, diff in enumerate(differences):
        print(f"   GELU diff[{i}]: {diff.item():.10f}")
    
    assert torch.allclose(gelu_output, expected_gelu, atol=1e-6), "GELU precision test failed"
    print("✅ GELU precision verified")

def compare_with_pytorch_gelu():
    """Compare exact GELU with PyTorch's built-in GELU"""
    print("\n🔄 Comparing with PyTorch built-in GELU")
    
    test_values = torch.tensor([0.2, -0.4, 1.0, 0.0], dtype=torch.float32)
    
    # Our exact implementation
    exact_gelu = gelu_exact(test_values)
    
    # PyTorch's exact GELU (same as our implementation)
    pytorch_exact = F.gelu(test_values, approximate='none')
    
    # PyTorch's tanh approximation
    pytorch_tanh = F.gelu(test_values, approximate='tanh')
    
    print(f"   Input: {test_values.tolist()}")
    print(f"   Our exact GELU: {exact_gelu.tolist()}")
    print(f"   PyTorch exact: {pytorch_exact.tolist()}")
    print(f"   PyTorch tanh approx: {pytorch_tanh.tolist()}")
    
    # Our implementation should match PyTorch's exact GELU
    assert torch.allclose(exact_gelu, pytorch_exact, atol=1e-7), "Our GELU doesn't match PyTorch exact"
    
    # Show difference between exact and tanh approximation
    diff_exact_tanh = torch.abs(pytorch_exact - pytorch_tanh)
    print(f"   Max diff (exact vs tanh): {diff_exact_tanh.max().item():.8f}")
    
    print("✅ GELU comparison verified - using exact erf formula")

if __name__ == "__main__":
    print("=" * 60)
    print("MLP Canonical Fixture Test (Python/PyTorch)")
    print("Cross-language verification with Rust/Burn implementation")
    print("=" * 60)
    
    try:
        # Run all tests
        test_gelu_precision()
        compare_with_pytorch_gelu()
        test_intermediate_values() 
        result = test_mlp_canonical_fixture()
        
        print(f"\n🎉 All tests passed!")
        print(f"Final canonical output: {result.tolist()}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
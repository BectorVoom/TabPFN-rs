"""
Python LayerNorm golden fixture tests using PyTorch.

This serves as the authoritative reference for numerical correctness
that the Rust implementation must match exactly within tolerance.
"""
import torch
import numpy as np
import pytest


def test_layernorm_golden_fixture():
    """Test LayerNorm against golden fixture values for exact numerical agreement."""
    torch.manual_seed(0)
    eps = 1e-5
    
    # Input tensor (batch=2, seq=3, d=4) - exact values from specification
    x = torch.tensor([
        [[1.0,  2.0,  3.0,  4.0],
         [0.5, -1.0,  0.0,  2.0],
         [10.0, 11.0, 12.0, 13.0]],

        [[-1.0, -2.0, -3.0, -4.0],
         [4.0,  0.0, -4.0,  2.0],
         [0.0,  0.0,  0.0,  0.0]]
    ], dtype=torch.float32)

    # Create LayerNorm with normalized_shape=[4], eps=1e-5
    # Ensure default initialization: weight=1, bias=0
    ln = torch.nn.LayerNorm(normalized_shape=[4], eps=eps, elementwise_affine=True)
    with torch.no_grad():
        if ln.weight is not None:
            ln.weight.fill_(1.0)
        if ln.bias is not None:
            ln.bias.zero_()

    # Apply LayerNorm
    y = ln(x)

    # Golden fixture expected values (computed using standard LayerNorm formula)
    expected = np.array([
        [[-1.3416355,  -0.44721183,   0.44721183,   1.3416355 ],
         [ 0.11546957, -1.2701652,   -0.3464087,    1.5011044 ],
         [-1.3416355,  -0.44721183,   0.44721183,   1.3416355 ]],
        [[ 1.3416355,   0.44721183,  -0.44721183,  -1.3416355 ],
         [ 1.1832154,  -0.16903076,  -1.5212768,    0.5070923 ],
         [ 0.0,         0.0,          0.0,          0.0 ]]
    ], dtype=np.float32)

    # Assert exact agreement within tolerance
    y_numpy = y.cpu().numpy()
    assert np.allclose(y_numpy, expected, atol=1e-6, rtol=1e-6), (
        f"LayerNorm output differs from golden fixture.\n"
        f"Got:\n{y_numpy}\n"
        f"Expected:\n{expected}\n"
        f"Max diff: {np.max(np.abs(y_numpy - expected))}"
    )

    print("✅ Golden fixture test passed - PyTorch LayerNorm produces expected values")


def test_layernorm_shape_preservation():
    """Test that LayerNorm preserves input tensor shape."""
    torch.manual_seed(42)
    
    # Test various shapes
    test_shapes = [
        (2, 3, 4),      # Original fixture shape
        (1, 5, 4),      # Single batch
        (3, 1, 4),      # Single sequence
        (1, 1, 4),      # Minimal shape
    ]
    
    for shape in test_shapes:
        x = torch.randn(shape, dtype=torch.float32)
        ln = torch.nn.LayerNorm(normalized_shape=[4], eps=1e-5)
        
        y = ln(x)
        
        assert y.shape == x.shape, (
            f"Shape not preserved: input {x.shape} -> output {y.shape}"
        )


def test_layernorm_mathematical_properties():
    """Test mathematical properties of LayerNorm output."""
    torch.manual_seed(123)
    
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    ln = torch.nn.LayerNorm(normalized_shape=[4], eps=1e-5)
    
    y = ln(x)
    
    # Check that normalized dimensions have approximately zero mean and unit variance
    # LayerNorm normalizes over the last dimension (d=4)
    mean_last_dim = torch.mean(y, dim=-1, keepdim=True)
    var_last_dim = torch.var(y, dim=-1, keepdim=True, unbiased=False)
    
    # Should be close to 0 mean and 1 variance (within numerical precision)
    assert torch.allclose(mean_last_dim, torch.zeros_like(mean_last_dim), atol=1e-6)
    assert torch.allclose(var_last_dim, torch.ones_like(var_last_dim), atol=1e-5)


if __name__ == "__main__":
    test_layernorm_golden_fixture()
    test_layernorm_shape_preservation()
    test_layernorm_mathematical_properties()
    print("All Python LayerNorm tests passed! ✅")
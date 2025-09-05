#!/usr/bin/env python3
"""
Python comparison script for TabPFN Transformer validation

This script creates equivalent synthetic inputs and demonstrates the expected
behavior for manual comparison with the Rust implementation.

Required packages: numpy, torch (install via uv on Serena MCP)

Usage on Serena MCP:
1. uv add numpy torch
2. uv run python python_comparison_transformer.py
"""

import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, List, Optional, Tuple
import sys

def create_synthetic_input(batch_size: int, seq_len: int, num_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic input data matching Rust test format
    
    Returns:
        x: Input tensor [seq_len, batch_size, num_features] dtype=float32
        y: Target tensor [0, batch_size, 1] dtype=float32 (empty for inference)
    """
    # Create input tensor with same values as Rust test (all 1.0)
    x = torch.ones(seq_len, batch_size, num_features, dtype=torch.float32)
    
    # Create empty target tensor (inference mode)
    y = torch.zeros(0, batch_size, 1, dtype=torch.float32)
    
    return x, y

def create_simple_transformer_like_model(emb_dim: int, num_features: int, n_out: int) -> nn.Module:
    """
    Create a simple transformer-like model for comparison
    
    This is a minimal reference that demonstrates the expected input/output shapes
    and basic operations similar to the Rust implementation.
    """
    class SimpleTransformer(nn.Module):
        def __init__(self, emb_dim: int, num_features: int, n_out: int):
            super().__init__()
            self.emb_dim = emb_dim
            self.num_features = num_features
            
            # Simple linear encoder (similar to SequentialEncoder)
            self.input_encoder = nn.Linear(1, emb_dim)  # features_per_group = 1
            self.y_encoder = nn.Linear(1, emb_dim)
            
            # Simple transformer-like layer
            self.transformer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=2,
                dim_feedforward=emb_dim,
                dropout=0.0,  # No dropout for reproducibility
                batch_first=False  # seq_first format
            )
            
            # Learned positional embedding (demonstration)
            self.feature_pos_emb = nn.Embedding(1000, emb_dim)
            
            # Output decoder
            self.decoder_linear1 = nn.Linear(emb_dim, emb_dim)
            self.decoder_linear2 = nn.Linear(emb_dim, n_out)
            
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Forward pass demonstrating expected behavior
            
            Args:
                x: [seq_len, batch_size, num_features]
                y: [seq_y, batch_size, 1]
            
            Returns:
                output: [batch_size, seq_len, n_out]
            """
            seq_len, batch_size, num_features = x.shape
            
            # Encode x: reshape to [seq_len * batch_size * num_features, 1] for linear layer
            x_flat = x.view(-1, 1)
            encoded_x = self.input_encoder(x_flat)  # -> [seq*batch*features, emb_dim]
            
            # Reshape back to [batch_size, seq_len, num_features, emb_dim]
            encoded_x = encoded_x.view(batch_size, seq_len, num_features, self.emb_dim)
            
            # Add positional embeddings
            feature_indices = torch.arange(num_features, dtype=torch.long)
            pos_embs = self.feature_pos_emb(feature_indices)  # [num_features, emb_dim]
            pos_embs = pos_embs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_features, emb_dim]
            encoded_x = encoded_x + pos_embs
            
            # Handle y (targets) - create dummy embedding for concatenation
            if y.numel() == 0:
                # Empty y for inference
                y_embedded = torch.zeros(batch_size, seq_len, self.emb_dim)
            else:
                # Encode y (similar process)
                y_flat = y.view(-1, 1)
                y_embedded = self.y_encoder(y_flat)
                y_embedded = y_embedded.view(batch_size, seq_len, self.emb_dim)
            
            # Expand y_embedded to match concatenation: [batch, seq, 1, emb_dim]
            y_embedded = y_embedded.unsqueeze(2)
            
            # Concatenate x and y: [batch, seq, features+1, emb_dim]
            combined = torch.cat([encoded_x, y_embedded], dim=2)
            
            # Reshape for transformer: [seq_len, batch*(features+1), emb_dim]
            seq_len, batch_size, combined_features, emb_dim = combined.shape
            transformer_input = combined.permute(1, 0, 2, 3).reshape(seq_len, batch_size * combined_features, emb_dim)
            
            # Apply transformer
            transformer_output = self.transformer(transformer_input)
            
            # Reshape back and extract target outputs
            transformer_output = transformer_output.view(seq_len, batch_size, combined_features, emb_dim)
            transformer_output = transformer_output.permute(1, 0, 2, 3)  # [batch, seq, features+1, emb]
            
            # Extract target outputs (last feature dimension)
            target_outputs = transformer_output[:, :, -1, :]  # [batch, seq, emb_dim]
            
            # Apply decoder
            hidden = torch.relu(self.decoder_linear1(target_outputs))
            output = self.decoder_linear2(hidden)  # [batch, seq, n_out]
            
            return output
    
    return SimpleTransformer(emb_dim, num_features, n_out)

def compare_with_rust_format():
    """
    Create test case that matches Rust test format exactly
    """
    print("=== TabPFN Transformer Python Reference Comparison ===")
    print()
    
    # Test parameters matching Rust tests
    batch_size = 2
    seq_len = 3
    num_features = 4
    emb_dim = 8
    n_out = 2
    
    print(f"Test Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  num_features: {num_features}")
    print(f"  emb_dim: {emb_dim}")
    print(f"  n_out: {n_out}")
    print()
    
    # Create synthetic input
    x, y = create_synthetic_input(batch_size, seq_len, num_features)
    
    print(f"Input shapes:")
    print(f"  x: {list(x.shape)} (dtype: {x.dtype})")
    print(f"  y: {list(y.shape)} (dtype: {y.dtype})")
    print()
    
    # Create model
    model = create_simple_transformer_like_model(emb_dim, num_features, n_out)
    model.eval()  # Set to evaluation mode
    
    # Set deterministic behavior
    torch.manual_seed(42)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, y)
    
    print(f"Output shape: {list(output.shape)} (dtype: {output.dtype})")
    print(f"Expected shape: [batch_size, seq_len, n_out] = [2, 3, 2]")
    print()
    
    # Check for NaN values
    has_nan = torch.isnan(output).any().item()
    print(f"Contains NaN: {has_nan}")
    
    # Basic statistics
    output_np = output.numpy()
    print(f"Output statistics:")
    print(f"  Mean: {output_np.mean():.6f}")
    print(f"  Std: {output_np.std():.6f}")
    print(f"  Min: {output_np.min():.6f}")
    print(f"  Max: {output_np.max():.6f}")
    print(f"  Non-zero elements: {np.count_nonzero(np.abs(output_np) > 1e-6)}/{output_np.size}")
    print()
    
    # Save outputs for comparison
    results = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_features": num_features,
            "emb_dim": emb_dim,
            "n_out": n_out,
            "seed": 42
        },
        "input_shapes": {
            "x": list(x.shape),
            "y": list(y.shape)
        },
        "output_shape": list(output.shape),
        "output_stats": {
            "mean": float(output_np.mean()),
            "std": float(output_np.std()),
            "min": float(output_np.min()),
            "max": float(output_np.max()),
            "has_nan": has_nan,
            "non_zero_count": int(np.count_nonzero(np.abs(output_np) > 1e-6))
        },
        "sample_output": output_np[:1, :1, :].tolist()  # First sample for comparison
    }
    
    # Save to file for comparison
    with open("python_transformer_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: python_transformer_comparison.json")
    print()
    print("To compare with Rust implementation:")
    print("1. Run the Rust tests: cargo test -- --nocapture")
    print("2. Compare output shapes and basic statistics")
    print("3. Verify both implementations handle the same input format")
    print("4. Check that neither produces NaN values")
    print()
    
    return results

def test_reproducibility():
    """Test that the same seed produces identical results"""
    print("=== Testing Reproducibility ===")
    
    batch_size, seq_len, num_features = 2, 3, 4
    emb_dim, n_out = 8, 2
    
    # First run
    torch.manual_seed(42)
    model1 = create_simple_transformer_like_model(emb_dim, num_features, n_out)
    x1, y1 = create_synthetic_input(batch_size, seq_len, num_features)
    
    model1.eval()
    with torch.no_grad():
        output1 = model1(x1, y1)
    
    # Second run with same seed
    torch.manual_seed(42)
    model2 = create_simple_transformer_like_model(emb_dim, num_features, n_out)
    x2, y2 = create_synthetic_input(batch_size, seq_len, num_features)
    
    model2.eval()
    with torch.no_grad():
        output2 = model2(x2, y2)
    
    # Check reproducibility
    diff = torch.abs(output1 - output2).max().item()
    print(f"Max difference between runs: {diff:.2e}")
    print(f"Reproducible (diff < 1e-6): {diff < 1e-6}")
    print()

def demonstrate_dtype_consistency():
    """Demonstrate f32 dtype consistency"""
    print("=== Testing dtype Consistency ===")
    
    # Create various tensors and verify they're all float32
    x = torch.ones(2, 3, 4, dtype=torch.float32)
    y = torch.zeros(0, 2, 1, dtype=torch.float32)
    
    print(f"Input x dtype: {x.dtype}")
    print(f"Input y dtype: {y.dtype}")
    
    # Verify numpy conversion
    x_np = x.numpy()
    print(f"Numpy conversion dtype: {x_np.dtype}")
    print(f"All dtypes are float32: {all([x.dtype == torch.float32, y.dtype == torch.float32, x_np.dtype == np.float32])}")
    print()

if __name__ == "__main__":
    try:
        print("Python TabPFN Transformer Comparison")
        print("=====================================")
        print()
        
        # Check dependencies
        print(f"NumPy version: {np.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print()
        
        # Run comparisons
        demonstrate_dtype_consistency()
        test_reproducibility()
        results = compare_with_rust_format()
        
        print("✅ All Python reference tests completed successfully!")
        print()
        print("Next steps for comparison:")
        print("1. Run Rust implementation: cargo test test_shape_correctness -- --nocapture")
        print("2. Compare output shapes and statistics")
        print("3. Verify reproducibility behavior")
        print("4. Check JSON output file for detailed comparison")
        
    except Exception as e:
        print(f"❌ Error running Python comparison: {e}")
        sys.exit(1)
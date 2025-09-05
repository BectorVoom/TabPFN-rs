#!/usr/bin/env python3
"""
Python comparison script for TabPFN Per-Feature Transformer

This script provides a Python reference implementation to validate the Rust
implementation's output shapes and behavior. It should be executed on a 
Serena MCP server using the uv package manager.

Setup Instructions (execute on Serena MCP server):
1. Install required packages using uv:
   uv add torch numpy tabpfn
   
2. Ensure Python environment is activated:
   uv python pin 3.11  # or preferred version
   
3. Run this script:
   uv run python python_comparison.py

Requirements:
- batch=2, seq=3, features=4, n_out=2
- Deterministic behavior with seed=42
- Shape validation: [batch, seq, n_out]
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def set_seed(seed: int = 42):
    """Set deterministic seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimplifiedPerFeatureTransformer(nn.Module):
    """
    Simplified Python reference implementation of Per-Feature Transformer.
    
    This is a minimal version that matches the test requirements:
    - batch=2, seq=3, features=4, n_out=2
    - Deterministic with seed=42
    - Output shape [batch, seq, n_out]
    """
    
    def __init__(
        self,
        emsize: int = 32,
        nhead: int = 2,
        nlayers: int = 1,
        n_out: int = 2,
        features_per_group: int = 1,
        seed: int = 42
    ):
        super().__init__()
        self.emsize = emsize
        self.nhead = nhead
        self.nlayers = nlayers
        self.n_out = n_out
        self.features_per_group = features_per_group
        self.seed = seed
        
        # Set seed for deterministic initialization
        set_seed(seed)
        
        # Simple input embedding - converts features to embedding dimension
        self.input_projection = nn.Linear(features_per_group, emsize)
        
        # Y encoder - minimal implementation
        self.y_projection = nn.Linear(1, emsize)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emsize,
            nhead=nhead,
            dim_feedforward=emsize * 4,
            dropout=0.0,  # No dropout for deterministic behavior
            activation='gelu',
            batch_first=True  # Use batch_first=True for [batch, seq, features] format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Decoder layers
        self.decoder_linear1 = nn.Linear(emsize, emsize * 4)
        self.decoder_linear2 = nn.Linear(emsize * 4, n_out)
        
        # Initialize weights deterministically
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with deterministic values."""
        set_seed(self.seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, num_features]
            y: Target tensor of shape [0, batch_size, 1] (empty for test)
            
        Returns:
            Output tensor of shape [batch_size, seq_len, n_out]
        """
        # Transpose input from [seq_len, batch, features] to [batch, seq, features]
        x = x.transpose(0, 1)  # [batch, seq, features]
        
        # Handle feature grouping - pad to multiple of features_per_group if needed
        batch_size, seq_len, num_features = x.shape
        missing_to_next = (self.features_per_group - (num_features % self.features_per_group)) % self.features_per_group
        if missing_to_next > 0:
            padding = torch.zeros(batch_size, seq_len, missing_to_next)
            x = torch.cat([x, padding], dim=2)
        
        # Reshape for feature groups: [batch, seq, features] -> [batch, seq, n_groups, features_per_group]
        n_groups = x.shape[2] // self.features_per_group
        x = x.reshape(batch_size, seq_len, n_groups, self.features_per_group)
        
        # Project input features to embedding dimension
        # [batch, seq, n_groups, features_per_group] -> [batch, seq, n_groups, emsize]
        embedded_x = self.input_projection(x)
        
        # Handle y input (targets) - create minimal embedding
        if y is None or y.shape[0] == 0:
            # No targets provided - create zero embedding
            embedded_y = torch.zeros(batch_size, seq_len, 1, self.emsize)
        else:
            # Transpose y and embed
            y = y.transpose(0, 1)  # [batch, seq, 1]
            embedded_y = self.y_projection(y).unsqueeze(2)  # [batch, seq, 1, emsize]
        
        # Combine x and y embeddings: [batch, seq, n_groups+1, emsize]
        combined = torch.cat([embedded_x, embedded_y], dim=2)
        
        # Flatten for transformer: [batch, seq*(n_groups+1), emsize]
        flat_seq_len = seq_len * (n_groups + 1)
        transformer_input = combined.reshape(batch_size, flat_seq_len, self.emsize)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Reshape back: [batch, seq*(n_groups+1), emsize] -> [batch, seq, n_groups+1, emsize]
        reshaped_output = transformer_output.reshape(batch_size, seq_len, n_groups + 1, self.emsize)
        
        # Extract target outputs (last group corresponds to targets)
        target_outputs = reshaped_output[:, :, -1, :]  # [batch, seq, emsize]
        
        # Apply decoder layers
        hidden = torch.gelu(self.decoder_linear1(target_outputs))
        output = self.decoder_linear2(hidden)
        
        return output  # [batch, seq, n_out]


def create_test_data(seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic test data matching Rust test requirements."""
    set_seed(seed)
    
    # Test requirements: batch=2, seq=3, features=4
    BATCH_SIZE = 2
    SEQ_LEN = 3
    NUM_FEATURES = 4
    
    # Create deterministic input data
    x_data = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_FEATURES)
    
    # Create empty y tensor (no targets for this test)
    y_data = torch.zeros(0, BATCH_SIZE, 1)
    
    return x_data, y_data


def test_shape_and_determinism():
    """Test that matches the Rust test requirements."""
    print("🔍 Testing Per-Feature Transformer Python Reference Implementation")
    print("=" * 60)
    
    # Test requirements
    BATCH_SIZE = 2
    SEQ_LEN = 3
    NUM_FEATURES = 4
    N_OUT = 2
    SEED = 42
    
    print(f"📊 Test Configuration:")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Sequence length: {SEQ_LEN}")
    print(f"   - Number of features: {NUM_FEATURES}")
    print(f"   - Output dimensions: {N_OUT}")
    print(f"   - Seed: {SEED}")
    print()
    
    # Create model
    model = SimplifiedPerFeatureTransformer(
        emsize=32,
        nhead=2,
        nlayers=1,
        n_out=N_OUT,
        features_per_group=1,
        seed=SEED
    )
    
    # Create test data
    x_data, y_data = create_test_data(SEED)
    
    print(f"📥 Input shapes:")
    print(f"   - x: {list(x_data.shape)} (seq_len, batch_size, num_features)")
    print(f"   - y: {list(y_data.shape)} (empty targets)")
    print()
    
    # First forward pass
    model.eval()  # Ensure deterministic behavior
    with torch.no_grad():
        output1 = model(x_data, y_data)
    
    # Verify output shape
    expected_shape = [BATCH_SIZE, SEQ_LEN, N_OUT]
    actual_shape = list(output1.shape)
    
    print(f"📤 Output verification:")
    print(f"   - Expected shape: {expected_shape}")
    print(f"   - Actual shape: {actual_shape}")
    print(f"   - Shape match: {'✅' if actual_shape == expected_shape else '❌'}")
    print()
    
    assert actual_shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
    
    # Second forward pass (should be identical)
    with torch.no_grad():
        output2 = model(x_data, y_data)
    
    # Check determinism
    diff = torch.abs(output1 - output2).max().item()
    is_deterministic = diff < 1e-6
    
    print(f"🔄 Determinism verification:")
    print(f"   - Max difference between runs: {diff:.2e}")
    print(f"   - Deterministic: {'✅' if is_deterministic else '❌'}")
    print()
    
    assert is_deterministic, f"Non-deterministic behavior detected: max diff = {diff}"
    
    print("✅ All tests passed!")
    print(f"   - Output shape: {actual_shape}")
    print(f"   - Deterministic behavior confirmed")
    print(f"   - Ready for Rust comparison")
    
    return output1.numpy()


def print_setup_instructions():
    """Print setup instructions for Serena MCP server."""
    print("🚀 Setup Instructions for Serena MCP Server")
    print("=" * 50)
    print()
    print("1. Install required packages using uv:")
    print("   uv add torch numpy")
    print("   # Optional: uv add tabpfn  # if using full TabPFN implementation")
    print()
    print("2. Set Python version (if needed):")
    print("   uv python pin 3.11")
    print()
    print("3. Run this comparison script:")
    print("   uv run python python_comparison.py")
    print()
    print("4. For Rust comparison:")
    print("   cargo test --lib test_transformer_module_trait -- --nocapture")
    print("   cargo test --lib test_transformer_shape_requirements -- --nocapture")
    print()
    print("📋 Expected Results:")
    print("   - Python output shape: [2, 3, 2]")
    print("   - Rust output shape: [2, 3, 2]")
    print("   - Both should be deterministic with seed=42")
    print()


if __name__ == "__main__":
    print_setup_instructions()
    
    print("\n" + "=" * 60)
    print("🧪 Running Tests")
    print("=" * 60)
    
    try:
        output = test_shape_and_determinism()
        
        print("\n" + "=" * 60)
        print("📊 Output Summary")
        print("=" * 60)
        print(f"Final output shape: {output.shape}")
        print(f"Output statistics:")
        print(f"  - Mean: {output.mean():.6f}")
        print(f"  - Std:  {output.std():.6f}")
        print(f"  - Min:  {output.min():.6f}")
        print(f"  - Max:  {output.max():.6f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
    
    print("\n🎯 Python reference implementation completed successfully!")
    print("Ready for cross-language comparison with Rust implementation.")
#!/usr/bin/env python3
"""
TabPFN-like Python Reference Implementation

This script demonstrates a small reference forward pass similar to TabPFN.
Run on Serena MCP server and use `uv add <package>` (no `pip`).

Setup commands for Serena MCP server:
    uv add torch
    uv add numpy  
    uv add scipy
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh


class SimpleTabPFNReference(nn.Module):
    """Simplified TabPFN-like model for reference comparison with Rust implementation"""
    
    def __init__(self, emsize=8, num_features=4, n_out=2, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.emsize = emsize
        self.num_features = num_features
        self.n_out = n_out
        
        # Simple linear encoder (instead of complex sequential encoder)
        self.feature_encoder = nn.Linear(1, emsize)
        self.target_encoder = nn.Linear(1, emsize)
        
        # Learned positional embeddings
        self.feature_pos_embeddings = nn.Embedding(num_features, emsize)
        
        # Simple transformer-like processing
        self.attention = nn.MultiheadAttention(emsize, num_heads=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(emsize)
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(emsize, emsize * 2),
            nn.GELU(),
            nn.Linear(emsize * 2, n_out)
        )
    
    def add_feature_positional_embeddings(self, x):
        """Add learnable positional embeddings to features"""
        batch_size, seq_len, num_features, emb_dim = x.shape
        
        # Create feature indices: [0, 1, 2, ..., num_features-1]
        feature_indices = torch.arange(num_features, device=x.device)
        pos_embs = self.feature_pos_embeddings(feature_indices)  # [num_features, emb_dim]
        
        # Broadcast to match x shape: [batch, seq, num_features, emb_dim]
        pos_embs = pos_embs.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        return x + pos_embs
    
    def create_dag_positional_encoding(self, batch_size, seq_len, emb_dim, dag_dim=2):
        """Create simple DAG-like positional encoding using spectral methods"""
        # Create a simple graph adjacency matrix (2 nodes: 0->1)
        adj_matrix = np.array([[0, 1], [0, 0]], dtype=np.float32)
        
        # Compute Laplacian
        degree_matrix = np.diag(adj_matrix.sum(axis=1))
        laplacian = degree_matrix - adj_matrix
        
        # Add small regularization for stability
        laplacian += 1e-6 * np.eye(2)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = eigh(laplacian)
        
        # Take smallest non-zero eigenvalues
        sorted_indices = np.argsort(eigenvals)
        selected_eigenvecs = eigenvecs[:, sorted_indices[1:1+dag_dim]]  # Skip first eigenvalue
        
        # Create positional encoding tensor
        dag_pos_enc = torch.from_numpy(selected_eigenvecs.astype(np.float32))
        
        # Apply to first batch item only (simplified)
        delta_x = torch.zeros(batch_size, seq_len, self.num_features, emb_dim)
        delta_y = torch.zeros(batch_size, seq_len, emb_dim)
        
        if batch_size > 0:
            # Apply DAG encoding to first 2 features and first dag_dim embedding dimensions
            features_to_use = min(2, self.num_features)
            dims_to_use = min(dag_dim, emb_dim)
            
            for i in range(features_to_use):
                delta_x[0, :, i, :dims_to_use] = dag_pos_enc[i, :dims_to_use]
            
            # Apply to target (simplified)
            delta_y[0, :, :dims_to_use] = dag_pos_enc[0, :dims_to_use]  # Use first node for target
        
        return delta_x, delta_y
    
    def forward(self, x, y=None, use_dag=False):
        """
        Forward pass similar to PerFeatureTransformer
        
        Args:
            x: Input features [batch, seq, num_features]
            y: Target values [batch, seq, 1] or None
            use_dag: Whether to apply DAG positional encoding
        """
        batch_size, seq_len, num_features = x.shape
        
        # Encode features: [batch, seq, num_features] -> [batch, seq, num_features, emb_dim]
        x_encoded = self.feature_encoder(x.unsqueeze(-1))  # Add feature dim for linear layer
        
        # Add learned positional embeddings
        x_encoded = self.add_feature_positional_embeddings(x_encoded)
        
        # Handle targets
        if y is None:
            y = torch.zeros(batch_size, seq_len, 1)
        
        y_encoded = self.target_encoder(y)  # [batch, seq, 1, emb_dim]
        
        # Apply DAG positional encoding if requested
        if use_dag:
            delta_x, delta_y = self.create_dag_positional_encoding(
                batch_size, seq_len, self.emsize, dag_dim=2
            )
            x_encoded = x_encoded + delta_x
            y_encoded = y_encoded + delta_y
        
        # Combine features and targets: [batch, seq, num_features+1, emb_dim]
        combined = torch.cat([x_encoded, y_encoded], dim=2)
        
        # Reshape for attention: [batch, seq*(num_features+1), emb_dim]
        batch_size, seq_len, num_tokens, emb_dim = combined.shape
        combined_flat = combined.reshape(batch_size, seq_len * num_tokens, emb_dim)
        
        # Apply attention
        attended, _ = self.attention(combined_flat, combined_flat, combined_flat)
        attended = self.layer_norm(attended + combined_flat)
        
        # Extract target tokens (last token of each sequence position)
        target_indices = torch.arange(num_features, seq_len * num_tokens, num_tokens + 1)
        target_outputs = attended[:, target_indices, :]  # [batch, seq, emb_dim]
        
        # Decode to final output
        output = self.decoder(target_outputs)  # [batch, seq, n_out]
        
        return output


def test_deterministic_behavior():
    """Test that same seeds produce identical outputs"""
    print("Testing deterministic behavior...")
    
    # Create identical models with same seed
    model1 = SimpleTabPFNReference(emsize=8, num_features=4, n_out=2, seed=42)
    model2 = SimpleTabPFNReference(emsize=8, num_features=4, n_out=2, seed=42)
    
    # Create test input
    batch_size, seq_len, num_features = 2, 3, 4
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.zeros(batch_size, seq_len, 1)
    
    # Forward pass with both models
    with torch.no_grad():
        out1 = model1(x, y)
        out2 = model2(x, y)
    
    # Check if outputs are identical
    diff = torch.abs(out1 - out2).max().item()
    print(f"Max difference between identical models: {diff}")
    assert diff < 1e-6, "Models with same seed should produce identical outputs"
    
    print("✅ Deterministic behavior test passed!")


def test_learned_embeddings():
    """Test that learned embeddings affect output"""
    print("\nTesting learned embedding contribution...")
    
    model = SimpleTabPFNReference(emsize=8, num_features=4, n_out=2, seed=42)
    
    # Create test input
    batch_size, seq_len, num_features = 2, 3, 4
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.zeros(batch_size, seq_len, 1)
    
    # Forward pass with embeddings
    with torch.no_grad():
        out_with = model(x, y)
    
    # Zero the embedding weights
    with torch.no_grad():
        model.feature_pos_embeddings.weight.zero_()
    
    # Forward pass without embeddings
    with torch.no_grad():
        out_without = model(x, y)
    
    # Check if outputs differ
    diff = torch.abs(out_with - out_without).max().item()
    print(f"Max difference with/without embeddings: {diff}")
    assert diff > 1e-6, "Learned embeddings should affect output"
    
    print("✅ Learned embedding test passed!")


def test_dag_encoding():
    """Test that DAG encoding affects output"""
    print("\nTesting DAG positional encoding...")
    
    model = SimpleTabPFNReference(emsize=8, num_features=4, n_out=2, seed=42)
    
    # Create test input
    batch_size, seq_len, num_features = 2, 3, 4
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.zeros(batch_size, seq_len, 1)
    
    # Forward pass without DAG
    with torch.no_grad():
        out_without_dag = model(x, y, use_dag=False)
    
    # Forward pass with DAG
    with torch.no_grad():
        out_with_dag = model(x, y, use_dag=True)
    
    # Check if outputs differ
    diff = torch.abs(out_without_dag - out_with_dag).max().item()
    print(f"Max difference with/without DAG: {diff}")
    assert diff > 1e-6, "DAG encoding should affect output"
    
    print("✅ DAG encoding test passed!")


def test_shapes():
    """Test that shapes are correct"""
    print("\nTesting shape correctness...")
    
    model = SimpleTabPFNReference(emsize=8, num_features=4, n_out=2, seed=42)
    
    # Create test input with exact specification dimensions
    batch_size, seq_len, num_features = 2, 3, 4
    x = torch.randn(batch_size, seq_len, num_features)
    y = torch.zeros(batch_size, seq_len, 1)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, y)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, 2)  # [batch, seq, n_out]
    actual_shape = output.shape
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")
    assert actual_shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"
    
    print("✅ Shape test passed!")


def main():
    """Run all tests to validate Python reference implementation"""
    print("TabPFN-like Python Reference Implementation")
    print("=" * 50)
    print("Run on Serena MCP server and use `uv add <package>` (no `pip`)")
    print("")
    
    test_deterministic_behavior()
    test_learned_embeddings() 
    test_dag_encoding()
    test_shapes()
    
    print("\n" + "=" * 50)
    print("🎉 All Python reference tests passed!")
    print("This validates the expected behavior for the Rust implementation.")


if __name__ == "__main__":
    main()
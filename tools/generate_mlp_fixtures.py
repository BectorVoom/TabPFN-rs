#!/usr/bin/env python3
"""
Generate MLP fixtures for Rust parity testing.

This script creates deterministic test fixtures using the Python TabPFN reference
implementation to ensure Rust implementation matches exactly with canonical values.

Implements the canonical test case specified in the user requirements:
- Input: [1.0, 2.0, -1.0]
- MLP: Linear(3→4, no bias) → GELU → Linear(4→3, no bias)
- Expected output: [0.11585194, -0.13783130, 0.84134475]
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Add the TabPFN source to path
script_dir = Path(__file__).parent
tabpfn_src = script_dir.parent / "TabPFN" / "src"
sys.path.insert(0, str(tabpfn_src))

try:
    from tabpfn.architectures.base.mlp import MLP, Activation
except ImportError as e:
    print(f"Error importing TabPFN modules: {e}")
    print("Please ensure the TabPFN Python implementation is available")
    sys.exit(1)


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_canonical_mlp_fixture() -> Dict[str, Any]:
    """
    Create the canonical MLP test fixture with exact values from user specification.
    
    Network specification:
    - Input: [1.0, 2.0, -1.0]
    - MLP: Linear(3→4, no bias) → GELU → Linear(4→3, no bias)
    - Weights as specified in canonical example
    - Expected output: [0.11585194, -0.13783130, 0.84134475]
    
    Returns:
        Dict containing all test data for Rust verification
    """
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Canonical input from specification
    input_data = torch.tensor([1.0, 2.0, -1.0], dtype=dtype, device=device)
    
    # Create MLP with exact architecture
    mlp = MLP(
        size=3,
        hidden_size=4,
        activation=Activation.GELU,
        device=device,
        dtype=dtype,
        initialize_output_to_zero=False,
        recompute=False
    )
    
    # Set exact weights from canonical specification
    # Note: PyTorch uses [out_features, in_features] layout
    linear1_weight = torch.tensor([
        [0.1, 0.2, 0.3],      # output neuron 0
        [0.0, -0.1, 0.2],     # output neuron 1  
        [0.5, 0.5, 0.5],      # output neuron 2
        [-0.2, 0.1, 0.0]      # output neuron 3
    ], dtype=dtype, device=device)  # shape: [4, 3]
    
    linear2_weight = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], # output neuron 0
        [0.0, 1.0, 0.0, 0.0], # output neuron 1
        [0.0, 0.0, 1.0, 0.0]  # output neuron 2
    ], dtype=dtype, device=device)  # shape: [3, 4]
    
    # Assign weights
    mlp.linear1.weight.data = linear1_weight
    mlp.linear2.weight.data = linear2_weight
    
    # Verify no biases (should be None)
    assert mlp.linear1.bias is None, "linear1 should have no bias"
    assert mlp.linear2.bias is None, "linear2 should have no bias"
    
    # Forward pass using exact GELU
    with torch.no_grad():
        # Expand input to batch size 1
        input_batch = input_data.unsqueeze(0)  # shape: [1, 3]
        
        # Forward pass
        output = mlp(input_batch)  # shape: [1, 3]
        output_flat = output.squeeze(0)  # shape: [3]
        
        # Manual verification of canonical computation
        print("=== Manual Forward Pass Verification ===")
        
        # Step 1: Linear1
        hidden = torch.matmul(input_data, linear1_weight.T)  # [3] @ [3,4] -> [4]
        print(f"Hidden after linear1: {hidden.numpy()}")
        expected_hidden = torch.tensor([0.2, -0.4, 1.0, 0.0], dtype=dtype)
        assert torch.allclose(hidden, expected_hidden, atol=1e-6), f"Hidden mismatch: {hidden} vs {expected_hidden}"
        
        # Step 2: GELU (exact erf formula)
        activated = torch.nn.functional.gelu(hidden, approximate='none')  # Use exact erf
        print(f"Hidden after GELU: {activated.numpy()}")
        expected_gelu = torch.tensor([0.11585194, -0.13783130, 0.84134475, 0.0], dtype=dtype)
        assert torch.allclose(activated, expected_gelu, atol=1e-6), f"GELU mismatch: {activated} vs {expected_gelu}"
        
        # Step 3: Linear2 (identity-like matrix picks first 3 elements)
        final_output = torch.matmul(activated, linear2_weight.T)  # [4] @ [4,3] -> [3]
        print(f"Final output: {final_output.numpy()}")
        expected_final = torch.tensor([0.11585194, -0.13783130, 0.84134475], dtype=dtype)
        assert torch.allclose(final_output, expected_final, atol=1e-6), f"Final output mismatch: {final_output} vs {expected_final}"
        
        # Verify against MLP output
        assert torch.allclose(output_flat, expected_final, atol=1e-6), f"MLP output mismatch: {output_flat} vs {expected_final}"
        
        print("✅ Manual verification passed!")
    
    # Create fixture data
    fixture = {
        "test_type": "mlp_canonical",
        "description": "Canonical MLP test case with exact values for cross-language verification",
        "input_shape": list(input_data.shape),
        "input": input_data.tolist(),
        "network": {
            "size": 3,
            "hidden_size": 4, 
            "activation": "GELU",
            "bias": False
        },
        "weights": {
            "linear1": linear1_weight.tolist(),  # [4, 3] - PyTorch layout
            "linear2": linear2_weight.tolist()   # [3, 4] - PyTorch layout
        },
        "expected_output": output_flat.tolist(),
        "tolerance": {
            "atol": 1e-6,
            "rtol": 1e-6
        },
        "metadata": {
            "pytorch_version": torch.__version__,
            "device": str(device),
            "dtype": str(dtype),
            "gelu_formula": "exact_erf",
            "weight_layout": "pytorch_out_in"
        }
    }
    
    return fixture


def create_additional_test_cases() -> List[Dict[str, Any]]:
    """Create additional test cases for comprehensive coverage."""
    test_cases = []
    
    # Test case 1: Different input values
    test_cases.append(create_mlp_test_case(
        input_vals=[0.5, -0.5, 1.5],
        size=3,
        hidden_size=4,
        test_name="mlp_test_case_1"
    ))
    
    # Test case 2: Different dimensions
    test_cases.append(create_mlp_test_case(
        input_vals=[2.0, -1.0],
        size=2,
        hidden_size=3,
        test_name="mlp_test_case_2"
    ))
    
    # Test case 3: ReLU activation
    test_cases.append(create_mlp_test_case(
        input_vals=[1.0, 2.0, -1.0],
        size=3,
        hidden_size=4,
        activation=Activation.RELU,
        test_name="mlp_relu_test"
    ))
    
    return test_cases


def create_mlp_test_case(
    input_vals: List[float], 
    size: int, 
    hidden_size: int,
    activation: Activation = Activation.GELU,
    test_name: str = "mlp_test",
    seed: int = 42
) -> Dict[str, Any]:
    """Create a single MLP test case with random weights."""
    set_seeds(seed)
    
    device = torch.device('cpu')
    dtype = torch.float32
    
    input_data = torch.tensor(input_vals, dtype=dtype, device=device)
    
    mlp = MLP(
        size=size,
        hidden_size=hidden_size,
        activation=activation,
        device=device,
        dtype=dtype,
        initialize_output_to_zero=False,
        recompute=False
    )
    
    # Forward pass
    with torch.no_grad():
        input_batch = input_data.unsqueeze(0)
        output = mlp(input_batch)
        output_flat = output.squeeze(0)
    
    return {
        "test_type": test_name,
        "input_shape": list(input_data.shape),
        "input": input_data.tolist(),
        "network": {
            "size": size,
            "hidden_size": hidden_size,
            "activation": activation.name,
            "bias": False
        },
        "weights": {
            "linear1": mlp.linear1.weight.data.tolist(),
            "linear2": mlp.linear2.weight.data.tolist()
        },
        "expected_output": output_flat.tolist(),
        "tolerance": {
            "atol": 1e-5,
            "rtol": 1e-5
        },
        "metadata": {
            "seed": seed,
            "device": str(device),
            "dtype": str(dtype)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Generate MLP fixtures for Rust testing')
    parser.add_argument('--output-dir', type=str, default='fixtures', 
                       help='Output directory for fixture files')
    parser.add_argument('--canonical-only', action='store_true',
                       help='Generate only the canonical test case')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Generating MLP test fixtures...")
    
    # Generate canonical fixture
    print("Creating canonical MLP fixture...")
    canonical_fixture = create_canonical_mlp_fixture()
    
    canonical_file = output_dir / "mlp_canonical.json"
    with open(canonical_file, 'w') as f:
        json.dump(canonical_fixture, f, indent=2)
    print(f"✅ Canonical fixture saved to {canonical_file}")
    
    if not args.canonical_only:
        # Generate additional test cases
        print("Creating additional test cases...")
        additional_cases = create_additional_test_cases()
        
        for i, test_case in enumerate(additional_cases):
            test_file = output_dir / f"mlp_test_{i+1}.json"
            with open(test_file, 'w') as f:
                json.dump(test_case, f, indent=2)
            print(f"✅ Test case {i+1} saved to {test_file}")
    
    print(f"\n🎉 All fixtures generated in {output_dir}")
    print(f"🔍 Canonical test case expected output: {canonical_fixture['expected_output']}")


if __name__ == "__main__":
    main()
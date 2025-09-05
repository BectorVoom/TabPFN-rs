#!/usr/bin/env python3
"""
Create canonical MLP fixture without external dependencies.

This creates the exact canonical test case specified in the user requirements:
- Input: [1.0, 2.0, -1.0]
- MLP: Linear(3→4, no bias) → GELU → Linear(4→3, no bias)
- Expected output: [0.11585194, -0.13783130, 0.84134475]
"""

import json
import math
from typing import Dict, List, Any


def erf(x: float) -> float:
    """Approximation of error function using Abramowitz and Stegun formula."""
    # Constants for the approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return sign * y


def gelu(x: float) -> float:
    """GELU activation using exact erf formula: x * 0.5 * (1 + erf(x/√2))"""
    return x * 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def create_canonical_fixture() -> Dict[str, Any]:
    """Create the canonical MLP test fixture with manually computed values."""
    
    # Canonical input from specification
    input_data = [1.0, 2.0, -1.0]
    
    # Canonical weights from specification
    # PyTorch layout: [out_features, in_features]
    linear1_weight = [
        [0.1, 0.2, 0.3],      # output neuron 0
        [0.0, -0.1, 0.2],     # output neuron 1  
        [0.5, 0.5, 0.5],      # output neuron 2
        [-0.2, 0.1, 0.0]      # output neuron 3
    ]  # shape: [4, 3]
    
    linear2_weight = [
        [1.0, 0.0, 0.0, 0.0], # output neuron 0
        [0.0, 1.0, 0.0, 0.0], # output neuron 1
        [0.0, 0.0, 1.0, 0.0]  # output neuron 2
    ]  # shape: [3, 4]
    
    print("=== Manual Forward Pass Computation ===")
    
    # Step 1: Linear1 - matrix multiplication
    # hidden = input @ weight.T
    hidden = []
    for i in range(4):  # 4 output neurons
        h = 0.0
        for j in range(3):  # 3 input features
            h += input_data[j] * linear1_weight[i][j]
        hidden.append(h)
    
    print(f"Input: {input_data}")
    print(f"Hidden after linear1: {hidden}")
    
    # Expected: [0.2, -0.4, 1.0, 0.0]
    expected_hidden = [0.2, -0.4, 1.0, 0.0]
    for i, (actual, expected) in enumerate(zip(hidden, expected_hidden)):
        assert abs(actual - expected) < 1e-6, f"Hidden[{i}] mismatch: {actual} vs {expected}"
    
    # Step 2: GELU activation
    activated = [gelu(h) for h in hidden]
    print(f"Hidden after GELU: {activated}")
    
    # Expected GELU values from specification
    expected_gelu = [0.11585194, -0.13783130, 0.84134475, 0.0]
    for i, (actual, expected) in enumerate(zip(activated, expected_gelu)):
        diff = abs(actual - expected)
        print(f"  GELU({hidden[i]}) = {actual}, expected = {expected}, diff = {diff}")
        assert diff < 1e-5, f"GELU[{i}] mismatch: {actual} vs {expected}"
    
    # Step 3: Linear2 - matrix multiplication
    # output = activated @ weight.T
    output = []
    for i in range(3):  # 3 output neurons
        o = 0.0
        for j in range(4):  # 4 hidden features
            o += activated[j] * linear2_weight[i][j]
        output.append(o)
    
    print(f"Final output: {output}")
    
    # Expected final output from specification
    expected_output = [0.11585194, -0.13783130, 0.84134475]
    for i, (actual, expected) in enumerate(zip(output, expected_output)):
        diff = abs(actual - expected)
        print(f"  Output[{i}] = {actual}, expected = {expected}, diff = {diff}")
        assert diff < 1e-5, f"Output[{i}] mismatch: {actual} vs {expected}"
    
    print("✅ Manual verification passed!")
    
    # Create fixture data
    fixture = {
        "test_type": "mlp_canonical",
        "description": "Canonical MLP test case with exact values for cross-language verification",
        "input_shape": [3],
        "input": input_data,
        "network": {
            "size": 3,
            "hidden_size": 4, 
            "activation": "GELU",
            "bias": False
        },
        "weights": {
            "linear1": linear1_weight,  # [4, 3] - PyTorch layout [out_features, in_features]
            "linear2": linear2_weight   # [3, 4] - PyTorch layout [out_features, in_features]
        },
        "expected_output": expected_output,
        "tolerance": {
            "atol": 1e-6,
            "rtol": 1e-6
        },
        "intermediate_values": {
            "hidden_after_linear1": hidden,
            "hidden_after_gelu": activated
        },
        "metadata": {
            "gelu_formula": "exact_erf",
            "weight_layout": "pytorch_out_in",
            "note": "Burn uses [in_features, out_features] layout - transpose needed"
        }
    }
    
    return fixture


def main():
    """Generate the canonical fixture and save to file."""
    print("Creating canonical MLP fixture...")
    
    fixture = create_canonical_fixture()
    
    # Create output directory
    import os
    os.makedirs("fixtures", exist_ok=True)
    
    # Save fixture
    output_file = "fixtures/mlp_canonical.json"
    with open(output_file, 'w') as f:
        json.dump(fixture, f, indent=2)
    
    print(f"✅ Canonical fixture saved to {output_file}")
    print(f"🔍 Expected output: {fixture['expected_output']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate transformer reference outputs for Rust parity testing.

This script creates deterministic test fixtures using the Python TabPFN reference
implementation to ensure Rust transformer implementation matches exactly.

Follows the exact JSON schema specified in the executive summary:
{
  "seed": 42,
  "num_classes": 2,
  "X_train": [[...], ...],   // shape: [N_train][F]
  "y_train": [0,1,...],      // shape: [N_train]
  "X_test": [[...], ...],    // shape: [N_test][F]
  "probs": [[...], ...]      // shape: [N_test][num_classes], probabilities (softmax)
}
"""

import os
import sys
import json
import numpy as np
import torch
import random
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add the TabPFN source to path
script_dir = Path(__file__).parent
tabpfn_src = script_dir.parent / "TabPFN" / "src"
sys.path.insert(0, str(tabpfn_src))

try:
    from tabpfn import TabPFNClassifier
except ImportError as e:
    print(f"Error importing TabPFN modules: {e}")
    print("Please ensure the TabPFN Python implementation is available")
    sys.exit(1)


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_reference_case(
    seed: int,
    n_train: int,
    n_test: int,
    n_features: int,
    num_classes: int,
    device: str = "cpu",
    case_id: str = None,
) -> Dict[str, Any]:
    """Generate a single transformer reference case."""
    
    set_seeds(seed)
    
    # Generate synthetic dataset
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(n_train,)).astype(np.int64)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    
    # Create TabPFN classifier with deterministic settings
    model = TabPFNClassifier(
        device=device, 
        random_state=seed,
        n_estimators=1,  # Use single estimator for determinism
        softmax_temperature=1.0,
        balance_probabilities=False,
        ignore_pretraining_limits=True,  # Allow more flexibility for testing
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Get probability predictions
    probs = model.predict_proba(X_test)
    
    reference = {
        "seed": seed,
        "num_classes": int(num_classes),
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(), 
        "probs": probs.tolist(),
    }
    
    # Add optional metadata
    if case_id:
        reference["case_id"] = case_id
        reference["metadata"] = {
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "device": device,
        }
    
    return reference


def generate_all_references(
    output_dir: Path, 
    num_cases: int = 4,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """Generate all transformer reference cases."""
    
    references = []
    
    # Case 1: Small binary classification
    references.append(generate_reference_case(
        seed=42,
        n_train=8,
        n_test=4,
        n_features=5,
        num_classes=2,
        device=device,
        case_id="small_binary",
    ))
    
    # Case 2: Medium multiclass
    references.append(generate_reference_case(
        seed=123,
        n_train=16,
        n_test=6,
        n_features=8,
        num_classes=3,
        device=device,
        case_id="medium_multiclass",
    ))
    
    # Case 3: Larger binary
    references.append(generate_reference_case(
        seed=456,
        n_train=32,
        n_test=8,
        n_features=12,
        num_classes=2,
        device=device,
        case_id="large_binary",
    ))
    
    # Case 4: Edge case - single feature
    references.append(generate_reference_case(
        seed=789,
        n_train=6,
        n_test=3,
        n_features=1,
        num_classes=2,
        device=device,
        case_id="single_feature",
    ))
    
    # Case 5: Many classes
    if num_cases > 4:
        references.append(generate_reference_case(
            seed=101112,
            n_train=20,
            n_test=5,
            n_features=6,
            num_classes=4,
            device=device,
            case_id="many_classes",
        ))
    
    # Case 6: High dimensional
    if num_cases > 5:
        references.append(generate_reference_case(
            seed=131415,
            n_train=24,
            n_test=6,
            n_features=20,
            num_classes=2,
            device=device,
            case_id="high_dimensional",
        ))
    
    return references[:num_cases]


def save_references(references: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save references to disk as JSON files."""
    
    # Create output directory
    reference_dir = output_dir / "tests" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual reference files
    for i, reference in enumerate(references):
        case_id = reference.get("case_id", f"case_{i+1}")
        filename = f"reference_{case_id}.json"
        filepath = reference_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(reference, f, indent=2)
        
        print(f"Generated reference {i+1}/{len(references)}: {case_id}")
    
    # Save consolidated reference file (matches the schema exactly)
    # Use the first case as the main reference for simple validation
    main_reference = references[0].copy()
    # Remove metadata for clean schema compliance
    if "case_id" in main_reference:
        del main_reference["case_id"]
    if "metadata" in main_reference:
        del main_reference["metadata"]
    
    main_reference_path = reference_dir / "reference_outputs.json"
    with open(main_reference_path, 'w') as f:
        json.dump(main_reference, f, indent=2)
    
    # Create manifest for all references
    manifest = {
        "format_version": "1.0",
        "description": "TabPFN transformer reference outputs for Rust parity testing",
        "schema": {
            "seed": "integer random seed used for generation",
            "num_classes": "number of classes in classification task",
            "X_train": "training features as nested array [N_train][F]",
            "y_train": "training labels as array [N_train]", 
            "X_test": "test features as nested array [N_test][F]",
            "probs": "predicted probabilities as nested array [N_test][num_classes]"
        },
        "tolerances": {
            "absolute": 1e-5,
            "relative": 1e-6,
            "note": "May need to use atol=1e-4 for float32/burn differences"
        },
        "total_cases": len(references),
        "main_reference": "reference_outputs.json",
        "cases": [
            {
                "case_id": ref.get("case_id", f"case_{i+1}"),
                "filename": f"reference_{ref.get('case_id', f'case_{i+1}')}.json",
                "seed": ref["seed"],
                "num_classes": ref["num_classes"],
                "shapes": {
                    "X_train": [len(ref["X_train"]), len(ref["X_train"][0]) if ref["X_train"] else 0],
                    "y_train": [len(ref["y_train"])],
                    "X_test": [len(ref["X_test"]), len(ref["X_test"][0]) if ref["X_test"] else 0],
                    "probs": [len(ref["probs"]), len(ref["probs"][0]) if ref["probs"] else 0],
                }
            }
            for i, ref in enumerate(references)
        ]
    }
    
    manifest_path = reference_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved {len(references)} references to {reference_dir}")
    print(f"Main reference: {main_reference_path}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TabPFN transformer reference outputs for Rust testing"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("."),
        help="Output directory for references (default: current directory)"
    )
    parser.add_argument(
        "--num-cases", 
        type=int, 
        default=4,
        help="Number of test cases to generate (default: 4)"
    )
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda"], 
        default="cpu",
        help="Device to use for computation (default: cpu)"
    )
    
    args = parser.parse_args()
    
    print("Generating TabPFN transformer reference outputs...")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of cases: {args.num_cases}")
    print(f"Device: {args.device}")
    
    # Check if CUDA is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Generate references
    references = generate_all_references(args.output_dir, args.num_cases, args.device)
    
    # Save to disk
    save_references(references, args.output_dir)
    
    print("✓ Reference generation completed successfully!")


if __name__ == "__main__":
    main()
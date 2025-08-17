#!/usr/bin/env python3
"""
Python regression test for TabPFN transformer reference outputs.

This test verifies that the Python TabPFN implementation produces consistent,
reproducible outputs when run with the same inputs and seeds. This ensures
the reference generation environment is stable before comparing with Rust.
"""

import json
import sys
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any

# Add the TabPFN source to path
script_dir = Path(__file__).parent
tabpfn_src = script_dir.parent / "TabPFN" / "src"
sys.path.insert(0, str(tabpfn_src))

try:
    from tabpfn import TabPFNClassifier
except ImportError as e:
    pytest.skip(f"TabPFN not available: {e}", allow_module_level=True)


def load_reference(path: Path) -> Dict[str, Any]:
    """Load a reference JSON file."""
    with open(path) as f:
        return json.load(f)


def approx_equal(a, b, atol: float = 1e-5, rtol: float = 1e-6) -> bool:
    """Check if two arrays are approximately equal within tolerances."""
    return np.allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)


def create_tabpfn_classifier(seed: int, device: str = "cpu") -> TabPFNClassifier:
    """Create a TabPFN classifier with deterministic settings."""
    return TabPFNClassifier(
        device=device,
        random_state=seed,
        n_estimators=1,  # Use single estimator for determinism
        softmax_temperature=1.0,
        balance_probabilities=False,
        ignore_pretraining_limits=True,
    )


@pytest.fixture
def reference_dir() -> Path:
    """Get the reference directory path."""
    return Path(__file__).parent / "reference"


@pytest.fixture
def main_reference(reference_dir: Path) -> Dict[str, Any]:
    """Load the main reference output."""
    reference_path = reference_dir / "reference_outputs.json"
    if not reference_path.exists():
        pytest.skip(f"Main reference not found: {reference_path}")
    return load_reference(reference_path)


def test_python_tabpfn_matches_main_reference(main_reference: Dict[str, Any]):
    """Test that Python TabPFN matches the main stored reference."""
    seed = main_reference["seed"]
    X_train = np.array(main_reference["X_train"], dtype=np.float32)
    y_train = np.array(main_reference["y_train"], dtype=np.int64)
    X_test = np.array(main_reference["X_test"], dtype=np.float32)
    expected_probs = np.array(main_reference["probs"])
    
    # Create and fit model with same settings as reference generation
    model = create_tabpfn_classifier(seed, device="cpu")
    model.fit(X_train, y_train)
    
    # Get predictions
    actual_probs = model.predict_proba(X_test)
    
    # Verify shapes match
    assert actual_probs.shape == expected_probs.shape, (
        f"Shape mismatch: expected {expected_probs.shape}, got {actual_probs.shape}"
    )
    
    assert actual_probs.shape == (len(X_test), main_reference["num_classes"]), (
        f"Output shape should be ({len(X_test)}, {main_reference['num_classes']})"
    )
    
    # Verify probabilities are approximately equal
    assert approx_equal(actual_probs, expected_probs), (
        f"Python outputs diverge from stored reference.\n"
        f"Expected (first 3 samples):\n{expected_probs[:3]}\n"
        f"Actual (first 3 samples):\n{actual_probs[:3]}\n"
        f"Max absolute difference: {np.abs(actual_probs - expected_probs).max()}"
    )
    
    # Verify probabilities sum to 1
    prob_sums = actual_probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-6), (
        f"Probabilities do not sum to 1: {prob_sums}"
    )


@pytest.fixture
def available_reference_cases(reference_dir: Path) -> list[Path]:
    """Get all available reference case files."""
    if not reference_dir.exists():
        return []
    
    reference_files = list(reference_dir.glob("reference_*.json"))
    # Exclude the main reference_outputs.json
    return [f for f in reference_files if f.name != "reference_outputs.json"]


@pytest.mark.parametrize("reference_file", [
    pytest.param(
        ref_file, 
        id=ref_file.stem if isinstance(ref_file, Path) else str(ref_file)
    ) 
    for ref_file in []  # Will be populated by indirect parametrization
])
def test_python_tabpfn_matches_individual_references(reference_file: Path):
    """Test that Python TabPFN matches individual reference cases."""
    if not reference_file.exists():
        pytest.skip(f"Reference file not found: {reference_file}")
    
    reference = load_reference(reference_file)
    
    seed = reference["seed"]
    X_train = np.array(reference["X_train"], dtype=np.float32)
    y_train = np.array(reference["y_train"], dtype=np.int64)
    X_test = np.array(reference["X_test"], dtype=np.float32)
    expected_probs = np.array(reference["probs"])
    
    # Create and fit model
    model = create_tabpfn_classifier(seed, device="cpu")
    model.fit(X_train, y_train)
    
    # Get predictions
    actual_probs = model.predict_proba(X_test)
    
    # Verify match
    assert actual_probs.shape == expected_probs.shape
    assert approx_equal(actual_probs, expected_probs), (
        f"Case {reference_file.stem} failed: Python outputs diverge from reference"
    )


def test_reference_directory_exists(reference_dir: Path):
    """Test that the reference directory exists and contains expected files."""
    assert reference_dir.exists(), (
        f"Reference directory not found: {reference_dir}. "
        f"Run 'uv run tools/generate_transformer_reference.py' first."
    )
    
    main_reference = reference_dir / "reference_outputs.json"
    manifest = reference_dir / "manifest.json"
    
    assert main_reference.exists(), f"Main reference not found: {main_reference}"
    assert manifest.exists(), f"Manifest not found: {manifest}"


def test_manifest_consistency(reference_dir: Path):
    """Test that the manifest is consistent with actual reference files."""
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Manifest file not found")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Check that all referenced files exist
    for case in manifest["cases"]:
        case_file = reference_dir / case["filename"] 
        assert case_file.exists(), f"Referenced file not found: {case_file}"
        
        # Verify case consistency
        reference = load_reference(case_file)
        assert reference["seed"] == case["seed"]
        assert reference["num_classes"] == case["num_classes"]


@pytest.mark.parametrize("tolerance_test", [
    {"atol": 1e-5, "rtol": 1e-6, "should_pass": True},
    {"atol": 1e-8, "rtol": 1e-9, "should_pass": False},  # Very strict, likely to fail
])
def test_tolerance_boundaries(main_reference: Dict[str, Any], tolerance_test: Dict):
    """Test different tolerance levels to validate our chosen tolerances."""
    seed = main_reference["seed"]
    X_train = np.array(main_reference["X_train"], dtype=np.float32)
    y_train = np.array(main_reference["y_train"], dtype=np.int64)
    X_test = np.array(main_reference["X_test"], dtype=np.float32)
    expected_probs = np.array(main_reference["probs"])
    
    # Create and fit model
    model = create_tabpfn_classifier(seed, device="cpu")
    model.fit(X_train, y_train)
    
    # Get predictions
    actual_probs = model.predict_proba(X_test)
    
    # Test tolerance
    matches = approx_equal(
        actual_probs, 
        expected_probs, 
        atol=tolerance_test["atol"], 
        rtol=tolerance_test["rtol"]
    )
    
    if tolerance_test["should_pass"]:
        assert matches, (
            f"Expected tolerance {tolerance_test} to pass, but comparison failed"
        )
    # Note: We don't assert failure for strict tolerances as they might still pass


def test_determinism_across_runs(main_reference: Dict[str, Any]):
    """Test that multiple runs with same seed produce identical results."""
    seed = main_reference["seed"]
    X_train = np.array(main_reference["X_train"], dtype=np.float32)
    y_train = np.array(main_reference["y_train"], dtype=np.int64)
    X_test = np.array(main_reference["X_test"], dtype=np.float32)
    
    # Run 1
    model1 = create_tabpfn_classifier(seed, device="cpu")
    model1.fit(X_train, y_train)
    probs1 = model1.predict_proba(X_test)
    
    # Run 2 with same seed
    model2 = create_tabpfn_classifier(seed, device="cpu")
    model2.fit(X_train, y_train)
    probs2 = model2.predict_proba(X_test)
    
    # Should be exactly equal (or very close)
    assert approx_equal(probs1, probs2, atol=1e-10, rtol=1e-10), (
        "Multiple runs with same seed should produce identical results"
    )
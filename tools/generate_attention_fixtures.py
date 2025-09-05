#!/usr/bin/env python3
"""
Generate attention fixtures for Rust parity testing.

This script creates deterministic test fixtures using the Python TabPFN reference
implementation to ensure Rust implementation matches exactly.
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
    from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
    from tabpfn.architectures.base.config import ModelConfig
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


def create_test_config(emsize: int = 64, nhead: int = 4) -> ModelConfig:
    """Create a test configuration for attention."""
    config = ModelConfig()
    config.emsize = emsize
    config.nhead = nhead
    config.attention_init_gain = 1.0
    config.recompute_attn = False
    return config


def generate_fixture_case(
    case_id: str,
    seed: int,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    emsize: int,
    nhead: int,
    d_k: int,
    d_v: int,
    share_kv_across_n_heads: int = 1,
    dropout_p: Optional[float] = None,
    use_self_attention: bool = True,
    cache_scenario: str = "none",  # "none", "cache_kv", "use_cached", "streaming"
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Generate a single test fixture case."""
    
    set_seeds(seed)
    torch_device = torch.device(device)
    
    config = create_test_config(emsize, nhead)
    
    # Create attention module
    attention = MultiHeadAttention(
        d_k=d_k,
        d_v=d_v,
        device=torch_device,
        dtype=dtype,
        config=config,
        share_kv_across_n_heads=share_kv_across_n_heads,
        dropout_p=dropout_p,
        softmax_scale=None,
        initialize_output_to_zero=False,
    )
    
    # Set to evaluation mode for deterministic behavior
    attention.eval()
    
    # Generate input tensors
    x = torch.randn(batch_size, seq_len_q, emsize, device=torch_device, dtype=dtype)
    
    if use_self_attention:
        x_kv = None
        actual_seq_len_kv = seq_len_q
    else:
        x_kv = torch.randn(batch_size, seq_len_kv, emsize, device=torch_device, dtype=dtype)
        actual_seq_len_kv = seq_len_kv
    
    fixture_data = {
        "case_id": case_id,
        "seed": seed,
        "config": {
            "batch_size": batch_size,
            "seq_len_q": seq_len_q,
            "seq_len_kv": actual_seq_len_kv,
            "emsize": emsize,
            "nhead": nhead,
            "d_k": d_k,
            "d_v": d_v,
            "share_kv_across_n_heads": share_kv_across_n_heads,
            "dropout_p": dropout_p,
            "use_self_attention": use_self_attention,
            "cache_scenario": cache_scenario,
        },
        "inputs": {
            "x": x.detach().cpu().numpy(),
        },
        "weights": {},
        "outputs": {},
        "cache_states": {},
    }
    
    if not use_self_attention:
        fixture_data["inputs"]["x_kv"] = x_kv.detach().cpu().numpy()
    
    # Extract weights
    if attention.w_q is not None:
        fixture_data["weights"]["w_q"] = attention.w_q.detach().cpu().numpy()
    if attention.w_k is not None:
        fixture_data["weights"]["w_k"] = attention.w_k.detach().cpu().numpy()
    if attention.w_v is not None:
        fixture_data["weights"]["w_v"] = attention.w_v.detach().cpu().numpy()
    if attention.w_kv is not None:
        fixture_data["weights"]["w_kv"] = attention.w_kv.detach().cpu().numpy()
    if attention.w_qkv is not None:
        fixture_data["weights"]["w_qkv"] = attention.w_qkv.detach().cpu().numpy()
    
    fixture_data["weights"]["w_out"] = attention.w_out.detach().cpu().numpy()
    
    # Generate outputs based on cache scenario
    with torch.no_grad():
        if cache_scenario == "none":
            # Simple forward pass without caching
            output = attention.forward(
                x,
                x_kv=x_kv,
                cache_kv=False,
                use_cached_kv=False,
            )
            fixture_data["outputs"]["output"] = output.detach().cpu().numpy()
            
        elif cache_scenario == "cache_kv":
            # Forward pass with KV caching
            output = attention.forward(
                x,
                x_kv=x_kv,
                cache_kv=True,
                use_cached_kv=False,
            )
            fixture_data["outputs"]["output_with_cache"] = output.detach().cpu().numpy()
            
            # Save cache state
            if attention._k_cache is not None:
                fixture_data["cache_states"]["k_cache"] = attention._k_cache.detach().cpu().numpy()
            if attention._v_cache is not None:
                fixture_data["cache_states"]["v_cache"] = attention._v_cache.detach().cpu().numpy()
            if attention._kv_cache is not None:
                fixture_data["cache_states"]["kv_cache"] = attention._kv_cache.detach().cpu().numpy()
                
        elif cache_scenario == "use_cached":
            # First, populate cache
            _ = attention.forward(
                x,
                x_kv=x_kv,
                cache_kv=True,
                use_cached_kv=False,
            )
            
            # Then use cached values (with new query)
            x_new_query = torch.randn_like(x)
            output_with_cached = attention.forward(
                x_new_query,
                x_kv=None,  # Should use cached
                cache_kv=False,
                use_cached_kv=True,
            )
            
            fixture_data["inputs"]["x_new_query"] = x_new_query.detach().cpu().numpy()
            fixture_data["outputs"]["output_with_cached_kv"] = output_with_cached.detach().cpu().numpy()
            
            # Save cache state used
            if attention._k_cache is not None:
                fixture_data["cache_states"]["k_cache_used"] = attention._k_cache.detach().cpu().numpy()
            if attention._v_cache is not None:
                fixture_data["cache_states"]["v_cache_used"] = attention._v_cache.detach().cpu().numpy()
            if attention._kv_cache is not None:
                fixture_data["cache_states"]["kv_cache_used"] = attention._kv_cache.detach().cpu().numpy()
                
        elif cache_scenario == "streaming":
            # Streaming scenario: process input in chunks
            chunk_size = max(1, seq_len_q // 2)
            
            # First chunk with caching
            x_chunk1 = x[:, :chunk_size, :]
            x_kv_chunk1 = x_kv[:, :chunk_size, :] if x_kv is not None else None
            
            output_chunk1 = attention.forward(
                x_chunk1,
                x_kv=x_kv_chunk1,
                cache_kv=True,
                use_cached_kv=False,
            )
            
            # Second chunk using cached values
            x_chunk2 = x[:, chunk_size:, :]
            x_kv_chunk2 = x_kv[:, chunk_size:, :] if x_kv is not None else None
            
            output_chunk2 = attention.forward(
                x_chunk2,
                x_kv=x_kv_chunk2,
                cache_kv=False,
                use_cached_kv=True,
            )
            
            # Reference: full sequence at once
            attention.empty_kv_cache()  # Clear cache
            output_full = attention.forward(
                x,
                x_kv=x_kv,
                cache_kv=False,
                use_cached_kv=False,
            )
            
            fixture_data["inputs"].update({
                "x_chunk1": x_chunk1.detach().cpu().numpy(),
                "x_chunk2": x_chunk2.detach().cpu().numpy(),
            })
            if not use_self_attention:
                fixture_data["inputs"].update({
                    "x_kv_chunk1": x_kv_chunk1.detach().cpu().numpy(),
                    "x_kv_chunk2": x_kv_chunk2.detach().cpu().numpy(),
                })
            
            fixture_data["outputs"].update({
                "output_chunk1": output_chunk1.detach().cpu().numpy(),
                "output_chunk2": output_chunk2.detach().cpu().numpy(),
                "output_full_reference": output_full.detach().cpu().numpy(),
            })
            
            fixture_data["config"]["chunk_size"] = chunk_size
    
    return fixture_data


def generate_all_fixtures(output_dir: Path, num_cases: int = 8) -> List[Dict[str, Any]]:
    """Generate all test fixtures."""
    
    fixtures = []
    
    # Case 1: Basic self-attention, small dimensions
    fixtures.append(generate_fixture_case(
        case_id="basic_self_attn_small",
        seed=42,
        batch_size=2,
        seq_len_q=4,
        seq_len_kv=4,
        emsize=32,
        nhead=4,
        d_k=8,
        d_v=8,
        use_self_attention=True,
        cache_scenario="none",
    ))
    
    # Case 2: Cross-attention
    fixtures.append(generate_fixture_case(
        case_id="cross_attention",
        seed=123,
        batch_size=1,
        seq_len_q=6,
        seq_len_kv=8,
        emsize=64,
        nhead=4,
        d_k=16,
        d_v=16,
        use_self_attention=False,
        cache_scenario="none",
    ))
    
    # Case 3: Different d_k and d_v
    fixtures.append(generate_fixture_case(
        case_id="different_dk_dv",
        seed=456,
        batch_size=2,
        seq_len_q=4,
        seq_len_kv=4,
        emsize=48,
        nhead=3,
        d_k=12,
        d_v=20,
        use_self_attention=True,
        cache_scenario="none",
    ))
    
    # Case 4: KV sharing across heads
    fixtures.append(generate_fixture_case(
        case_id="kv_sharing",
        seed=789,
        batch_size=1,
        seq_len_q=4,
        seq_len_kv=4,
        emsize=64,
        nhead=8,
        d_k=8,
        d_v=8,
        share_kv_across_n_heads=2,
        use_self_attention=True,
        cache_scenario="none",
    ))
    
    # Case 5: With caching
    fixtures.append(generate_fixture_case(
        case_id="with_caching",
        seed=101112,
        batch_size=1,
        seq_len_q=6,
        seq_len_kv=6,
        emsize=32,
        nhead=4,
        d_k=8,
        d_v=8,
        use_self_attention=True,
        cache_scenario="cache_kv",
    ))
    
    # Case 6: Using cached KV
    fixtures.append(generate_fixture_case(
        case_id="use_cached_kv",
        seed=131415,
        batch_size=1,
        seq_len_q=4,
        seq_len_kv=4,
        emsize=32,
        nhead=2,
        d_k=16,
        d_v=16,
        use_self_attention=True,
        cache_scenario="use_cached",
    ))
    
    # Case 7: Streaming scenario
    fixtures.append(generate_fixture_case(
        case_id="streaming",
        seed=161718,
        batch_size=1,
        seq_len_q=8,
        seq_len_kv=8,
        emsize=48,
        nhead=4,
        d_k=12,
        d_v=12,
        use_self_attention=True,
        cache_scenario="streaming",
    ))
    
    # Case 8: Large batch, dropout
    fixtures.append(generate_fixture_case(
        case_id="large_batch_dropout",
        seed=192021,
        batch_size=4,
        seq_len_q=3,
        seq_len_kv=3,
        emsize=64,
        nhead=8,
        d_k=8,
        d_v=8,
        dropout_p=0.1,
        use_self_attention=True,
        cache_scenario="none",
    ))
    
    return fixtures[:num_cases]


def save_fixtures(fixtures: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save fixtures to disk as .npz files with manifest."""
    
    # Create output directories
    fixtures_dir = output_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "format_version": "1.0",
        "description": "TabPFN MultiHeadAttention test fixtures for Rust parity testing",
        "generation_date": str(torch.utils.data.get_worker_info()),
        "total_cases": len(fixtures),
        "cases": []
    }
    
    for i, fixture in enumerate(fixtures):
        case_id = fixture["case_id"]
        filename = f"{case_id}.npz"
        filepath = fixtures_dir / filename
        
        # Prepare data for saving
        save_data = {}
        
        # Flatten nested dictionaries for npz format
        for key, value in fixture["inputs"].items():
            save_data[f"input_{key}"] = value
            
        for key, value in fixture["weights"].items():
            save_data[f"weight_{key}"] = value
            
        for key, value in fixture["outputs"].items():
            save_data[f"output_{key}"] = value
            
        for key, value in fixture["cache_states"].items():
            save_data[f"cache_{key}"] = value
        
        # Save as npz
        np.savez_compressed(filepath, **save_data)
        
        # Add to manifest
        case_manifest = {
            "case_id": case_id,
            "filename": filename,
            "seed": fixture["seed"],
            "config": fixture["config"],
            "input_keys": list(fixture["inputs"].keys()),
            "weight_keys": list(fixture["weights"].keys()),
            "output_keys": list(fixture["outputs"].keys()),
            "cache_keys": list(fixture["cache_states"].keys()),
        }
        manifest["cases"].append(case_manifest)
        
        print(f"Generated fixture {i+1}/{len(fixtures)}: {case_id}")
    
    # Save manifest
    manifest_path = fixtures_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Saved {len(fixtures)} fixtures to {fixtures_dir}")
    print(f"Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate attention fixtures for Rust testing")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory for fixtures (default: current directory)")
    parser.add_argument("--num-cases", type=int, default=8,
                        help="Number of test cases to generate (default: 8)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device to use for computation (default: cpu)")
    
    args = parser.parse_args()
    
    print("Generating TabPFN MultiHeadAttention test fixtures...")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of cases: {args.num_cases}")
    print(f"Device: {args.device}")
    
    # Check if CUDA is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Generate fixtures
    fixtures = generate_all_fixtures(args.output_dir, args.num_cases)
    
    # Save to disk
    save_fixtures(fixtures, args.output_dir)
    
    print("✓ Fixture generation completed successfully!")


if __name__ == "__main__":
    main()
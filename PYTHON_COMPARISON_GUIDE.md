# Python Comparison Guide for TabPFN-rs Transformer

This guide provides instructions for running the Python reference implementation to validate the Rust Per-Feature Transformer implementation.

## 🎯 Purpose

This comparison validates that the Rust implementation produces:
- Correct output shapes: `[batch=2, seq=3, n_out=2]`
- Deterministic behavior with seed=42
- Equivalent computational behavior to Python reference

## 🖥️ Execution Environment

**IMPORTANT**: All Python operations must be executed on a **Serena MCP server** using the `uv` package manager, not `pip`.

## 📋 Setup Instructions

### Step 1: Install Required Packages (on Serena MCP Server)

```bash
# Install PyTorch and NumPy using uv (NOT pip)
uv add torch numpy

# Optional: Install full TabPFN if needed
# uv add tabpfn
```

### Step 2: Set Python Version

```bash
# Pin Python version (recommended)
uv python pin 3.11
```

### Step 3: Verify Installation

```bash
# Check that packages are available
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## 🧪 Running the Comparison

### Execute Python Reference

```bash
# Run the Python comparison script on Serena MCP server
uv run python python_comparison.py
```

### Execute Rust Tests

```bash
# Run Rust tests for comparison
cargo test --lib test_transformer_module_trait -- --nocapture
cargo test --lib test_transformer_shape_requirements -- --nocapture
```

## 📊 Expected Results

### Python Reference Output
```
🔍 Testing Per-Feature Transformer Python Reference Implementation
============================================================

📊 Test Configuration:
   - Batch size: 2
   - Sequence length: 3
   - Number of features: 4
   - Output dimensions: 2
   - Seed: 42

📥 Input shapes:
   - x: [3, 2, 4] (seq_len, batch_size, num_features)
   - y: [0, 2, 1] (empty targets)

📤 Output verification:
   - Expected shape: [2, 3, 2]
   - Actual shape: [2, 3, 2]
   - Shape match: ✅

🔄 Determinism verification:
   - Max difference between runs: 0.00e+00
   - Deterministic: ✅

✅ All tests passed!
   - Output shape: [2, 3, 2]
   - Deterministic behavior confirmed
   - Ready for Rust comparison
```

### Rust Test Output
```
✅ Module trait and deterministic behavior test passed!
   - Module trait properly implemented
   - Ignored fields accessible
   - Seed determinism verified
   - Constructor creates valid instances

✅ Shape requirements test passed!
   - Input shape: [3, 2, 4]
   - Expected output shape: [2, 3, 2]
   - Transformer configured correctly
```

## 🔍 Key Validation Points

1. **Shape Consistency**: Both implementations should produce `[2, 3, 2]` output shape
2. **Deterministic Behavior**: Repeated runs with seed=42 should yield identical results
3. **Module Trait**: Rust implementation properly derives `Module` trait
4. **F32 Consistency**: All computations use f32 precision
5. **Ignored Fields**: Non-module fields properly wrapped with `Ignored<T>`

## 🛠️ Troubleshooting

### Common Issues

1. **Package Installation Failed**
   ```bash
   # Ensure you're using uv, not pip
   uv add torch numpy
   # NOT: pip install torch numpy
   ```

2. **Python Version Conflicts**
   ```bash
   # Pin specific Python version
   uv python pin 3.11
   ```

3. **Import Errors**
   ```bash
   # Verify packages are in uv environment
   uv run python -c "import torch, numpy"
   ```

4. **Non-Deterministic Results**
   - Ensure both implementations use the same seed (42)
   - Check that dropout is disabled in eval mode
   - Verify CUDA deterministic settings

### Debugging Commands

```bash
# Check uv environment
uv run python --version
uv run python -c "import sys; print(sys.path)"

# Verify package versions
uv run python -c "import torch; print(torch.__version__)"

# Test basic functionality
uv run python -c "import torch; x = torch.randn(2,3,4); print(x.shape)"
```

## 📁 File Structure

```
TabPFN-rs/
├── python_comparison.py          # Python reference implementation
├── PYTHON_COMPARISON_GUIDE.md    # This guide
├── src/tabpfn/architectures/base/
│   └── transformer.rs            # Rust implementation with tests
└── Cargo.toml                    # Rust dependencies
```

## 🔗 Related Files

- **Rust Implementation**: `src/tabpfn/architectures/base/transformer.rs`
- **Rust Tests**: Functions `test_transformer_module_trait()` and `test_transformer_shape_requirements()`
- **Python Reference**: `python_comparison.py`

## ✅ Success Criteria

The comparison is successful when:

1. ✅ Python script runs without errors on Serena MCP server
2. ✅ Both implementations produce `[2, 3, 2]` output shape
3. ✅ Both show deterministic behavior (identical repeated runs)
4. ✅ Rust tests pass with proper Module trait implementation
5. ✅ All dependencies installed via `uv` (not `pip`)

## 📞 Support

If you encounter issues:

1. Verify you're running on Serena MCP server
2. Ensure using `uv` instead of `pip`
3. Check Python and package versions
4. Verify Rust compilation succeeds with `cargo check`
5. Compare output shapes and determinism between implementations
# PerFeatureTransformer Implementation

A complete Rust implementation of the PerFeatureTransformer using the Burn deep learning framework.

## Features

✅ **Module Derive & Parameter Management**: Proper `#[derive(Module)]` with `Ignored<T>` fields  
✅ **Deterministic RNG System**: Comprehensive deterministic random number generation with `DeterministicRngContext`  
✅ **Learned Embeddings**: Full `nn::Embedding` implementation with integer index lookup  
✅ **DAG Positional Encoding**: Spectral embeddings using `nalgebra` f32 with device-safe operations  
✅ **Device-Safe NaN Detection**: No CPU synchronization, uses `tensor.is_nan().any()`  
✅ **Shape Safety**: Proper const-generic tensor dimensions throughout  

## Commands

### Compilation Check
```bash
cargo check
```
**Expected Result**: Should compile without errors or warnings.

### Run All Tests
```bash
cargo test
```
**Expected Result**: All 6 tests (A-F) should pass, validating:
- Test A: RNG isolation & deterministic reproducibility
- Test B: Learned embedding contribution verification  
- Test C: DAG positional embedding application validation
- Test D: Device-safe NaN detection testing
- Test E: Module derive & Ignored fields correctness
- Test F: Shape & dtype verification

### Run Specific Tests
```bash
# Test RNG isolation and reproducibility
cargo test rng_isolation_and_reproducibility

# Test learned embeddings
cargo test learned_embedding_contributes

# Test DAG positional embeddings
cargo test dag_embedding_applied

# Test NaN detection
cargo test nan_detection_device_side

# Test Module trait implementation
cargo test module_derive_and_ignored_correctness

# Test shape safety
cargo test shape_and_dtype_checks
```

### Verbose Test Output
```bash
cargo test -- --nocapture
```
Shows detailed test output including print statements.

### Documentation Generation
```bash
cargo doc --open
```
Generates and opens documentation for the implementation.

## Expected Test Behavior

1. **RNG Tests**: Same seeds produce identical outputs, different seeds produce different outputs
2. **Embedding Tests**: Learned embeddings affect model output, different configurations produce different results  
3. **DAG Tests**: DAG positional encodings modify outputs when provided
4. **NaN Tests**: Device-side NaN detection catches invalid inputs without CPU sync
5. **Module Tests**: Proper parameter registration and `Ignored` field accessibility
6. **Shape Tests**: Correct tensor dimensions throughout the pipeline

## Deterministic RNG System

This implementation uses a comprehensive deterministic random number generation system to ensure full reproducibility across different runs and backends.

### Core Components

**DeterministicRngContext<B: Backend>**: The central RNG context that manages all random operations:
- Wraps a base seed and target device
- Provides deterministic tensor generation methods
- Ensures reproducible initialization and forward passes
- Never uses global RNG sources like `StdRng::from_entropy()` or `thread_rng()`

### Seed Offset Policy

All random operations use a deterministic seed offset schedule:
- **Base seed**: `config.seed` (primary model seed)
- **+100 series**: Parameter initialization (`linear1=+100`, `linear2=+101`, etc.)
- **+200 series**: Embedding initialization (`feature_pos_emb=+200`, `compression=+300`) 
- **+1000 series**: Forward pass randomness (`layer_dropout=+1000`, `pos_emb_gen=+2000`)

Example seed offset usage:
```rust
// Parameter initialization with offset +100
let rng_ctx = DeterministicRngContext::new(config.seed, device);
let linear_layer = rng_ctx.create_deterministic_linear(input_dim, output_dim, true, config.seed + 100);

// Forward pass randomness with offset +1000  
let output = rng_ctx.with_isolated_seed(Some(config.seed + 1000), |rng| {
    model.transformer_forward(inputs, targets, true, Some(rng), None, None, None)
});
```

### Constructor Requirements

All parameter-holding modules must accept `&DeterministicRngContext<Backend>`:
```rust
pub fn new(
    config: &ModelConfig,
    rng_ctx: &DeterministicRngContext<B>,  // Required for all modules with parameters
    init_seed_offset: u64,                  // Unique offset for this module
    // ... other parameters
) -> Self
```

### Forward Method Requirements

Modules with randomness during forward pass must accept `rng: Option<&mut StdRng>`:
```rust
pub fn forward(
    &mut self,
    input: Tensor<B, N>,
    rng: Option<&mut StdRng>,  // Required for dropout, layer dropout, etc.
    // ... other parameters  
) -> Tensor<B, M>
```

### Type Alignment Rule (Critical)

The `DeterministicRngContext<B>` type parameter must match the backend of the module's `Param` tensors:
- If module stores `Param<Tensor<B, ...>>` → use `DeterministicRngContext<B>`
- If module stores `Param<Tensor<B::InnerBackend, ...>>` → use `DeterministicRngContext<B::InnerBackend>`

### Forbidden Patterns

Production code must never use:
- **Global RNG**: `StdRng::from_entropy()`, `rand::thread_rng()`, `thread_rng()`
- **CPU sync ops**: `.to_data()`, `.as_slice()`, `.into_data()`, `.to_vec()` (except in documented helpers)

### CPU Sync Operations Policy

**Forbidden in production forward paths:**
- `.to_data()` - Forces CPU synchronization
- `.as_slice()` - Requires CPU memory access  
- `.into_data()` - Converts tensor to CPU data
- `.to_vec()` - CPU vector conversion

**Allowed exceptions (documented and tested):**
- Test functions for verification and debugging
- `DeterministicEmbedding` helper for integer index lookups (isolated, tested)
- Static analysis and validation tests

**Preferred alternatives:**
- Use device-native tensor operations (`.slice()`, `.reshape()`, `.cat()`)
- Device-safe reductions (`.sum()`, `.mean()`, `.argmax()`) 
- In-place tensor operations that stay on device

### Adding New Modules

When implementing new modules:
1. Accept `rng_ctx: &DeterministicRngContext<Backend>` in constructor
2. Use `rng_ctx.generate_normal_tensor()` or `rng_ctx.create_deterministic_linear()` for parameters
3. Add `rng: Option<&mut StdRng>` to forward methods that need randomness
4. Use unique seed offsets for each parameter and randomness source
5. Ensure `#[derive(Module)]` and proper `Param<Tensor<...>>` field registration

## Performance Notes

- Uses `burn-ndarray` backend for CPU computation during testing
- GPU backends (`burn-wgpu`, `burn-cuda`) can be enabled via features
- Device-safe operations avoid expensive CPU synchronization
- Deterministic RNG ensures reproducible results across runs

## Implementation Status

All major functional requirements have been implemented:
- ✅ No `unimplemented!()` or `todo!()` macros in critical paths
- ✅ Full tensor construction instead of unsupported slice assignment  
- ✅ Proper f32 dtype consistency with nalgebra operations
- ✅ Complete deterministic RNG system with comprehensive seed management
- ✅ All constructors updated to accept `DeterministicRngContext`
- ✅ All forward methods updated to accept optional RNG for reproducible randomness
- ✅ Global RNG usage eliminated from production code
- ✅ Blocking tests for deterministic behavior (parameter registration, init, forward, RNG/CPU sync checks)
- ✅ Complete unit test coverage for all specified requirements
# Minimal Cargo.toml Configuration for TabPFN Per-Feature Transformer

This document provides the minimal `Cargo.toml` snippet required for the Per-Feature Transformer implementation.

## Required Dependencies

```toml
[package]
name = "tabpfn-transformer"
version = "0.1.0"
edition = "2024"  # or "2021" if preferred

[dependencies]
# Core Burn framework with required features
burn = { version = "0.18.0", features = ["wgpu", "cuda", "autodiff", "train"] }

# Burn backend support
burn-ndarray = "0.18.0"
burn-wgpu = { version = "0.18.0", optional = true }
burn-cuda = { version = "0.18.0", optional = true }
burn-autodiff = "0.18.0"

# Linear algebra for DAG positional encoding
nalgebra = "0.33.0"

# Random number generation (deterministic)
rand = "0.8.5"

# Graph operations for DAG processing
petgraph = "0.6.5"

# Serialization for configuration
serde = { version = "1.0", features = ["derive"] }

# Standard library additions
num-traits = "0.2"

[features]
default = ["wgpu"]
wgpu = ["burn-wgpu"]
cuda = ["burn-cuda"]

[dev-dependencies]
# Testing utilities
approx = "0.5"
```

## Feature Selection

### Default Backend
- **WGPU**: Cross-platform GPU acceleration via WebGPU
- Provides good performance on most systems

### Optional Backends
- **CUDA**: NVIDIA GPU acceleration (requires CUDA toolkit)
- **NdArray**: CPU-only fallback (always available)

### Required Features
- **autodiff**: Automatic differentiation for training
- **train**: Training utilities and optimizers

## Backend Configuration

Choose backend based on your target environment:

```toml
# For CPU-only environments
[features]
default = []

# For NVIDIA GPU systems
[features]
default = ["cuda"]

# For general GPU acceleration
[features]
default = ["wgpu"]
```

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| **Burn** | 0.18.0 | Core deep learning framework |
| **Rust** | 1.70+ | Edition 2024 support |
| **CUDA** | 11.8+ | If using CUDA backend |

## Minimal Example Usage

```rust
use burn::{
    module::Module,
    nn,
    tensor::{backend::Backend, Tensor},
};

// Your transformer implementation here
#[derive(Module, Debug)]
pub struct PerFeatureTransformer<B: Backend> {
    // Module components automatically registered
    encoder: nn::Linear<B>,
    
    // Non-module fields wrapped with Ignored
    config: burn::module::Ignored<ModelConfig>,
}
```

## Build Commands

```bash
# Check compilation
cargo check

# Run tests
cargo test

# Build with specific features
cargo build --features cuda
cargo build --features wgpu
cargo build --no-default-features  # CPU only
```

This minimal configuration provides everything needed for the Per-Feature Transformer implementation while maintaining flexibility for different deployment environments.
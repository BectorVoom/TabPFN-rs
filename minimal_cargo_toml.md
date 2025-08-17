# Minimal Cargo.toml Dependencies for PerFeatureTransformer

```toml
[package]
name = "transformer-implementation"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core Burn framework for deep learning
burn = { version = "0.18.0", features = ["autodiff"] }
burn-ndarray = "0.18.0"
burn-autodiff = "0.18.0"

# Linear algebra for DAG spectral embeddings  
nalgebra = "0.33.0"

# Random number generation for deterministic RNG policy
rand = "0.8.5"
rand_distr = "0.4.3"

# Graph operations for DAG processing
petgraph = "0.6.5"

# Serialization for configuration
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
# For testing and validation
approx = "0.5"
```

## Required Features

- **burn**: Core deep learning framework with autodiff support
- **burn-ndarray**: CPU backend for testing and validation
- **burn-autodiff**: Automatic differentiation capabilities
- **nalgebra**: Linear algebra operations for spectral embeddings (using f32)
- **rand/rand_distr**: Deterministic random number generation with CPU_SEED policy  
- **petgraph**: Graph data structures for DAG operations
- **serde**: Configuration serialization support

## Optional Backend Features

For GPU acceleration, you can also include:

```toml
burn-wgpu = { version = "0.18.0", optional = true }  # WebGPU backend
burn-cuda = { version = "0.18.0", optional = true }  # CUDA backend

[features]
default = []
wgpu = ["burn-wgpu"]
cuda = ["burn-cuda"]
```
# TabPFN-rs Project Overview

## Project Purpose
TabPFN-rs is a Rust implementation of TabPFN (Tabular Prior-data Fitted Networks), which is a foundation model specifically designed for tabular data. This project is a port from the original Python implementation, focusing on providing high-performance neural network architectures for tabular data processing.

## Tech Stack
- **Language**: Rust (Edition 2024)
- **Deep Learning Framework**: Burn v0.18.0 with features for WGPU, CUDA, and autodiff
- **Key Dependencies**:
  - `burn` - Core deep learning framework
  - `burn-ndarray`, `burn-wgpu`, `burn-cuda`, `burn-autodiff` - Backend support
  - `polars` v0.50.0 - Data manipulation
  - `serde` - Serialization/deserialization
  - `tokio` - Async runtime
  - `petgraph` - Graph operations
  - `nalgebra` - Linear algebra
  - `rayon` - Parallelization
  - `rand` - Random number generation

## Project Status
- **Active Development**: This is an ongoing port from Python TabPFN
- **Full Attention Module**: ✅ Complete implementation with comprehensive testing
- **Multiple Test Binaries**: Extensive equivalence testing between Python and Rust implementations
- **Architecture Components**: Core modules for transformer, MLP, attention, encoders, and memory

## Features
- GPU acceleration support (WGPU, CUDA)
- Multiple backend support via Burn framework
- Comprehensive testing framework with equivalence validation
- Modular architecture following the original Python structure
- Configuration management with settings system

## Current Implementation Status
- ✅ Full attention mechanism with KV caching
- ✅ MLP (Multi-Layer Perceptron) components
- ✅ Core transformer architecture
- ✅ Settings and configuration system
- 🚧 Various components in active development (see unimplemented_tasks.md)

## Binary Targets
The project includes multiple test binaries for validating equivalence:
- `test_attention_equivalence` - Tests attention mechanism
- `test_mlp_equivalence` - Tests MLP components  
- `test_layer_equivalence` - Tests layer implementations
- `test_dropout_functionality` - Tests dropout behavior
- `debug_weights` - Weight debugging utilities
- And several other test/debug binaries
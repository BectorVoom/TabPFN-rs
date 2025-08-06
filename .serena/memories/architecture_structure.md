# TabPFN-rs Architecture Structure

## Top-Level Project Structure
```
TabPFN-rs/
├── src/
│   ├── lib.rs                 # Main library entry point
│   ├── main.rs               # Main application entry
│   ├── bin/                  # Test and debug binaries
│   └── tabpfn/               # Core TabPFN implementation
├── Cargo.toml               # Project configuration
└── Cargo.lock              # Dependency lock file
```

## Core Module Organization

### Library Structure (`src/lib.rs`)
- Exports the main `tabpfn` module
- Provides access to `Settings` and `settings` function
- Entry point: `pub use tabpfn::settings::{Settings, settings};`

### Main TabPFN Module (`src/tabpfn/`)
```
src/tabpfn/
├── mod.rs                   # Module exports (settings, architectures)
├── settings.rs              # Configuration management
└── architectures/           # Neural network architectures
```

## Architecture Components (`src/tabpfn/architectures/`)

### Module Structure
```
src/tabpfn/architectures/
├── mod.rs                   # Architecture module exports
├── interface.rs             # Architecture interfaces/traits
├── instance.rs              # Architecture instances
├── simple_instance.rs       # Simplified architecture instances
└── base/                    # Core neural network components
```

### Base Components (`src/tabpfn/architectures/base/`)
```
src/tabpfn/architectures/base/
├── mod.rs                   # Base module exports
├── config.rs                # Configuration structures
├── encoders.rs              # Data encoding components
├── layer.rs                 # Neural network layers
├── memory.rs                # Memory management utilities
├── mlp.rs                   # Multi-Layer Perceptron implementation
├── transformer.rs           # Transformer architecture
└── attention/               # Attention mechanisms
    ├── mod.rs
    └── full_attention.rs    # ✅ Complete multi-head attention
```

## Binary Targets (`src/bin/`)

### Testing Binaries
- `test_attention_equivalence.rs` - Validates attention implementation against Python
- `test_mlp_equivalence.rs` - Validates MLP implementation against Python  
- `test_layer_equivalence.rs` - Validates layer implementation against Python
- `test_dropout_functionality.rs` - Tests dropout behavior
- `test_forward_method_equivalence.rs` - Tests forward pass methods
- `test_cache_update_equivalence.rs` - Tests caching mechanisms
- `test_set_parameters.rs` - Tests parameter setting functionality
- `test_transformer_graph_operations.rs` - Tests transformer graph operations

### Debug Utilities
- `debug_weights.rs` - Weight inspection and debugging tools

## Key Architectural Patterns

### Configuration System
- **Settings Hierarchy**: `Settings` → `TabPFNSettings`, `TestingSettings`, `PytorchSettings`
- **Global Access**: Via `SETTINGS` constant and `settings()` function
- **Serialization**: Full serde support for all configuration structures

### Neural Network Components
- **Backend Abstraction**: Uses Burn framework with configurable backends (NdArray, WGPU, CUDA)
- **Module Pattern**: All components implement Burn's `Module` trait
- **Parameter Management**: Uses `Param<Tensor>` for trainable parameters

### Python Equivalence Structure
- **Direct Mapping**: Rust modules mirror Python file structure
- **Equivalence Testing**: Dedicated test binaries for each major component
- **Documentation Links**: Each Rust file references corresponding Python file

## Implementation Status

### ✅ Completed Components
- **Full Attention** (`full_attention.rs`) - Complete with comprehensive testing
- **MLP** (`mlp.rs`) - Multi-layer perceptron with activation functions
- **Settings System** (`settings.rs`) - Configuration management
- **Core Infrastructure** - Module organization and build system

### 🚧 In Development
- Various layer implementations
- Encoder components  
- Memory management utilities
- Transformer integration

## Data Flow Architecture

### Typical Processing Pipeline
1. **Configuration Loading** - Via settings system
2. **Data Preprocessing** - Through encoder components
3. **Neural Network Processing** - MLP, attention, transformer layers
4. **Output Generation** - Via appropriate output layers

### Backend Support
- **NdArray Backend** - CPU computation using ndarray
- **WGPU Backend** - GPU computation via WebGPU
- **CUDA Backend** - NVIDIA GPU acceleration
- **Autodiff Support** - Automatic differentiation for training

## Module Dependencies

### External Dependencies
- `burn` - Core deep learning framework
- `serde` - Configuration serialization
- `polars` - Data manipulation
- `tokio` - Async runtime support

### Internal Dependencies
- Settings system used throughout all modules
- Base components shared across architecture implementations
- Test utilities shared across equivalence testing binaries
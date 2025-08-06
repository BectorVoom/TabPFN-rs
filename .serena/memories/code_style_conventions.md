# Code Style and Conventions for TabPFN-rs

## File Headers and Copyright
All source files should include the copyright header:
```rust
//! Copyright (c) Prior Labs GmbH 2025.
//!
//! [Module description] - Rust implementation of
//! [corresponding Python file path]
```

## Documentation Style

### Module Documentation
Use `//!` for module-level documentation at the top of files:
```rust
//! Multi-Layer Perceptron (MLP) module - Rust implementation of
//! src/tabpfn/architectures/base/mlp.py
```

### Function/Struct Documentation
Use `///` for documenting public items:
```rust
/// Enum for activation functions - equivalent to Python Activation enum
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    GELU = 1,
    RELU = 2,
}
```

## Naming Conventions

### Structs and Enums
- PascalCase for struct and enum names: `MultiHeadAttention`, `TabPFNSettings`
- Enum variants in UPPERCASE: `GELU`, `RELU`

### Functions and Variables
- snake_case for functions and variables: `from_str`, `test_type`

### Constants
- UPPERCASE with underscores: `SETTINGS`

## Serialization/Deserialization
Consistently use Serde for configuration and data structures:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    // fields...
}
```

Common derives for data structures:
- `Debug, Clone` - Standard for most structs
- `PartialEq` - For enums and comparable types
- `Serialize, Deserialize` - For configuration and data transfer

## Error Handling
Use `Result` types for fallible operations:
```rust
pub fn from_str(s: &str) -> Result<Self, String> {
    match s.to_uppercase().as_str() {
        "GELU" => Ok(Activation::GELU),
        "RELU" => Ok(Activation::RELU),
        _ => Err(format!("Unknown activation function: {}", s)),
    }
}
```

## Import Organization
Group imports in this order:
1. Standard library imports
2. External crate imports (burn, serde, etc.)
3. Local crate imports

Example:
```rust
use std::env;
use std::fs;

use burn::prelude::*;
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};

use tab_pfn_rs::tabpfn::architectures::base::mlp::{Activation, MLP};
```

## Type Aliases
Use descriptive type aliases for complex types:
```rust
type Backend = NdArray<f32>;
```

## Test Structure
Test binaries should:
- Include comprehensive module documentation
- Use descriptive struct names for test data: `TestData`
- Implement proper deserialization for JSON test data
- Follow the pattern: `//! Test binary for [functionality] tests`

## Comments
- Use `//` for inline comments
- Provide context for Python equivalence: `// equivalent to Python's Activation[activation.upper()]`
- Explain Rust-specific implementation details when different from Python

## Configuration Management
- Use nested configuration structures (`TabPFNSettings`, `TestingSettings`, etc.)
- Implement `Default` trait for all configuration structs
- Use global configuration access patterns with `SETTINGS` constant

## Module Organization
- Follow the Python module structure closely
- Use `mod.rs` files to organize module exports
- Keep related functionality grouped in appropriate submodules
- Export public APIs at appropriate levels
# Tech Stack

## Python Side (Original)
- **Framework**: PyTorch 
- **Graph Operations**: NetworkX for directed acyclic graphs (DAGs)
- **Numerical**: NumPy for array operations
- **Linear Algebra**: SciPy for eigendecomposition
- **Tensor Operations**: einops for tensor reshaping
- **Other**: warnings, contextlib, functools, typing

## Rust Side (Target)
- **Framework**: Burn 0.18.0 (deep learning framework for Rust)
- **Features**: wgpu, cuda, autodiff, train
- **Graph Operations**: petgraph (replaces NetworkX)
- **Linear Algebra**: nalgebra for matrix operations  
- **Numerical**: Standard Rust numerics
- **Parallel Computing**: rayon
- **Serialization**: serde with JSON support
- **Random**: rand crate
- **Other**: polars, tokio, config, dirs, num-traits, log

## Key Differences
- PyTorch → Burn: Different tensor API and automatic differentiation system
- NetworkX → petgraph: Different graph representation and algorithms
- NumPy/SciPy → nalgebra: Different linear algebra approach
- Python dynamic typing → Rust static typing with generics
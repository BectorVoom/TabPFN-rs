# Essential Commands for TabPFN-rs Development

## Core Cargo Commands

### Building
```bash
# Build the project (default features: wgpu)
cargo build

# Build with CUDA support
cargo build --features cuda

# Build in release mode
cargo build --release
```

### Testing
```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test binary
cargo test --bin <binary_name>
```

### Running Binaries
```bash
# List all available binaries
cargo run --bin

# Run specific test binaries for equivalence testing
cargo run --bin test_mlp_equivalence
cargo run --bin test_attention_equivalence
cargo run --bin test_layer_equivalence
cargo run --bin test_dropout_functionality
cargo run --bin debug_weights
cargo run --bin test_forward_method_equivalence
cargo run --bin test_cache_update_equivalence
cargo run --bin test_set_parameters
cargo run --bin test_transformer_graph_operations

# Run main application
cargo run
```

### Development & Debugging
```bash
# Check code without building
cargo check

# Format code
cargo fmt

# Run clippy linter
cargo clippy

# Generate documentation
cargo doc --open

# Clean build artifacts
cargo clean
```

## macOS/Darwin Specific Commands
Since this project runs on Darwin (macOS), standard Unix commands are available:

### File Operations
```bash
# List files
ls -la

# Find files
find . -name "*.rs" -type f

# Search in files (ripgrep recommended if available)
grep -r "pattern" src/
# or use rg if available
rg "pattern" src/

# Navigate directories
cd src/tabpfn/architectures/
```

### Git Operations
```bash
# Check status
git status

# View changes
git diff

# Add and commit changes
git add .
git commit -m "message"
```

## Project-Specific Workflows

### Equivalence Testing
The project includes comprehensive equivalence tests to validate Rust implementation against Python:
```bash
# Run all equivalence tests
cargo run --bin test_mlp_equivalence
cargo run --bin test_attention_equivalence
cargo run --bin test_layer_equivalence
```

### Weight Debugging
```bash
# Debug weight loading and tensor operations
cargo run --bin debug_weights
```

### Development Cycle
1. Make code changes
2. `cargo check` - Quick syntax/type check
3. `cargo test` - Run tests
4. `cargo run --bin <relevant_test>` - Test specific functionality
5. `cargo clippy` - Check for common issues
6. `cargo fmt` - Format code
# Development Commands

## Build and Compilation
- `cargo build` - Build the Rust project
- `cargo check` - Fast compilation check without producing binary
- `cargo test` - Run Rust tests

## Python Development (for comparison/testing)
- `pip install -e "TabPFN[dev]"` - Install Python version in development mode
- `pytest tests/` - Run Python tests
- `python -m uv` - Alternative to pip for package management

## Code Quality
- `cargo fmt` - Format Rust code
- `cargo clippy` - Rust linter
- `pre-commit run --all-files` - Run Python pre-commit hooks (if working with Python side)

## Testing Equivalence
The project requires continuous equivalence testing between Python and Rust implementations:
- Use identical input values for both implementations
- Compare outputs at each major step of computation
- Run tests throughout development, not just at completion

## Binary Targets
Currently commented out in Cargo.toml due to compilation errors:
- test_mlp_equivalence
- debug_weights  
- test_attention_equivalence
- test_dropout_functionality
- test_set_parameters
- test_layer_equivalence
- test_transformer_graph_operations

These will be re-enabled once architecture modules are fixed.

## Development Workflow
1. Build with `cargo build` at every iteration
2. Run equivalence tests continuously during development
3. Verify semantic equivalence between Python and Rust implementations
4. Ensure compilation success is maintained throughout
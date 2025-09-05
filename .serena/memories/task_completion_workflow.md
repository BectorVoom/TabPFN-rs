# Task Completion Workflow for TabPFN-rs

## Standard Development Workflow

### 1. Code Development Phase
When implementing new features or fixing bugs:

1. **Understand the Python equivalent** (if porting functionality)
2. **Write Rust implementation** following project conventions
3. **Add appropriate documentation** with `//!` and `///` comments
4. **Include error handling** where appropriate

### 2. Testing and Validation

#### Compilation Check
Always ensure code compiles before proceeding:
```bash
cargo check
```

#### Unit Testing
Run relevant tests to verify functionality:
```bash
# Run all tests
cargo test

# Run specific test if available
cargo test [test_name]
```

#### Equivalence Testing
For ported functionality, run equivalence tests against Python implementation:
```bash
# Run appropriate equivalence test binary
cargo run --bin test_mlp_equivalence          # For MLP changes
cargo run --bin test_attention_equivalence    # For attention changes
cargo run --bin test_layer_equivalence        # For layer changes
cargo run --bin test_dropout_functionality    # For dropout changes
```

### 3. Code Quality Checks

#### Linting
Run Clippy to catch common issues:
```bash
cargo clippy
```

#### Formatting
Format code according to project standards:
```bash
cargo fmt
```

### 4. Final Verification

#### Build Verification
Ensure final build succeeds:
```bash
cargo build
```

#### Documentation Generation
Verify documentation builds correctly:
```bash
cargo doc
```

## Specific Testing Patterns

### For Neural Network Components
1. Test basic forward pass functionality
2. Test parameter loading and initialization
3. Test equivalence with Python implementation
4. Test edge cases and error conditions

### For Configuration Changes
1. Verify settings load correctly
2. Test default values
3. Test serialization/deserialization

### For New Binary Targets
1. Ensure proper command-line argument handling
2. Test JSON data loading (for equivalence tests)
3. Verify output format matches expected structure

## Debugging Workflow

### Weight and Tensor Issues
Use the debug utilities:
```bash
cargo run --bin debug_weights
```

### Equivalence Test Failures
1. Run specific equivalence test with detailed output
2. Compare tensor shapes and values
3. Check parameter loading
4. Verify computation paths

### Performance Issues
1. Use release build for performance testing: `cargo build --release`
2. Profile with appropriate tools
3. Compare with Python baseline performance

## Git Workflow Integration

### Before Committing
1. Run full test suite: `cargo test`
2. Run clippy: `cargo clippy`
3. Format code: `cargo fmt`
4. Verify build: `cargo build`
5. Run relevant equivalence tests

### Commit Message Format
Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `test:` for test additions
- `docs:` for documentation
- `refactor:` for code refactoring

## Continuous Integration Mindset
Always assume changes will be tested automatically:
- Ensure all tests pass locally before pushing
- Maintain backward compatibility where possible
- Document breaking changes clearly
- Keep commits focused and atomic

## Project-Specific Considerations

### Python Equivalence Priority
- Always verify equivalence with Python implementation when porting
- Document any intentional deviations from Python behavior
- Maintain comprehensive test coverage for ported functionality

### Performance Expectations
- Rust implementation should match or exceed Python performance
- GPU acceleration should work correctly across backends (WGPU, CUDA)
- Memory usage should be optimized appropriately
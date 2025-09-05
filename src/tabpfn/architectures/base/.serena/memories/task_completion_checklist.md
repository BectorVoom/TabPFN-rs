# Task Completion Checklist

When working on any task in this project, ensure these steps are completed:

## Compilation Verification
- [ ] `cargo build` completes successfully without errors
- [ ] `cargo check` passes clean
- [ ] No warnings that indicate semantic issues

## Code Quality
- [ ] `cargo fmt` applied (if format changes needed)
- [ ] `cargo clippy` warnings addressed
- [ ] Code follows established patterns and conventions

## Semantic Equivalence Testing
- [ ] Identify equivalent Python functionality
- [ ] Create test cases with identical inputs
- [ ] Compare outputs between Python and Rust implementations  
- [ ] Verify intermediate computations match where possible
- [ ] Test edge cases and error conditions

## Documentation
- [ ] Public functions have appropriate documentation
- [ ] Complex algorithms include explanatory comments
- [ ] Any deviations from Python implementation are documented

## Integration Testing
- [ ] New code integrates with existing architecture
- [ ] Module interfaces remain consistent
- [ ] No breaking changes to dependent components

## Memory and Performance
- [ ] Caching mechanisms work correctly
- [ ] Device placement handled appropriately
- [ ] No memory leaks or excessive allocations

## Error Handling
- [ ] Appropriate error types and messages
- [ ] Graceful failure modes
- [ ] Consistent with Python error behavior where applicable

## Final Verification
- [ ] End-to-end test passes with realistic data
- [ ] Performance is reasonable compared to Python
- [ ] All functionality from Python version is preserved
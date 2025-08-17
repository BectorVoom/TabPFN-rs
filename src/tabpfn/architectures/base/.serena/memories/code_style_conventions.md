# Code Style and Conventions

## Rust Conventions
- **Edition**: 2024 (cutting edge)
- **Naming**: snake_case for functions/variables, PascalCase for types
- **Generics**: Use `<B: Backend>` pattern extensively for Burn backend abstraction
- **Error Handling**: Return `Result<T, String>` for operations that can fail
- **Documentation**: Use `///` for public API documentation
- **Module Structure**: `#[derive(Module, Debug)]` for neural network components
- **Skip Fields**: Use `#[module(skip)]` for non-trainable parameters in modules

## Python Equivalence Requirements
- **Function Names**: Preserve original Python function names exactly
- **Structure**: Maintain same module organization and class hierarchy
- **Behavior**: Ensure identical computational behavior and outputs
- **Parameters**: Keep same parameter names and default values where possible

## Burn Framework Patterns
- **Tensors**: Use `Tensor<B, D>` where B is backend, D is dimensionality
- **Modules**: Implement neural network components as modules with `#[derive(Module)]`
- **Forward Pass**: Use `.forward()` method for neural network layers
- **Device**: Pass device context for tensor creation
- **Activation**: Use `burn::tensor::activation` for activation functions

## Thread Safety
- Use `Arc<Mutex<T>>` for shared mutable state (cached embeddings)
- Implement `Send + Sync` where needed for multi-threading

## Graph Operations
- Use petgraph `Graph<NodeMetadata, (), Directed>` for DAG representation
- Maintain NodeMetadata structure with feature/target indices
- Implement graph algorithms to match NetworkX behavior exactly

## Memory Management
- Use explicit caching mechanisms with Optional types
- Clear caches appropriately in `empty_trainset_representation_cache`
- Handle device placement consistently
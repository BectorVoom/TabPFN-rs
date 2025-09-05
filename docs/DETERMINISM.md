# TabPFN-rs Determinism Documentation

This document describes the deterministic RNG implementation in TabPFN-rs and provides guidelines for maintaining deterministic behavior across all operations.

## 🎯 Overview

TabPFN-rs implements a fully deterministic neural network system that ensures:

1. **Reproducible parameter initialization** across different runs with the same seed
2. **Deterministic forward passes** with controllable stochastic operations
3. **Proper train/eval semantics** with explicit control over randomness
4. **Secure attention masking** to prevent label leakage
5. **Zero dependency on global RNG sources**

## 🏗️ Architecture

### DeterministicRngContext

The core of our deterministic system is the `DeterministicRngContext<B>` struct:

```rust
pub struct DeterministicRngContext<B: Backend> {
    pub seed: u64,
    device: B::Device,
    cache: Ignored<Arc<Mutex<Option<Tensor<B, 2>>>>>,
}
```

**Key Methods:**
- `with_isolated_seed()`: Creates isolated RNG for specific operations
- `create_deterministic_linear()`: Deterministic linear layer creation
- `create_deterministic_embedding()`: Deterministic embedding layer creation
- `create_deterministic_layer_norm()`: Deterministic layer normalization
- `generate_normal_tensor()`: Deterministic normal tensor generation
- `generate_uniform_tensor()`: Deterministic uniform tensor generation

### Seed Offset Policy

We use a structured seed offset system to ensure different components use different but deterministic random seeds:

```
Base seed (config.seed): Primary model seed
+100 series: Parameter initialization
  - +100: First linear layer
  - +101: Second linear layer
  - +102: Third linear layer, etc.

+200 series: Embedding initialization
  - +200: Feature positional embeddings
  - +300: Global attention embeddings
  - +400: Additional embeddings

+1000 series: Forward pass randomness
  - +1000: Layer dropout
  - +2000: Positional embedding generation
  - +3000: Additional forward randomness
```

## 🔧 Usage Patterns

### Model Creation

```rust
let rng_context = DeterministicRngContext::new(config.seed, device);
let model = PerFeatureTransformer::new(&config, &rng_context, ...);
```

### Parameter Initialization

```rust
// Replace this:
let linear = LinearConfig::new(input_dim, output_dim).init(device);

// With this:
let linear = rng_ctx.create_deterministic_linear::<B>(
    input_dim, 
    output_dim, 
    bias, 
    seed_offset
);
```

### Forward Pass with Determinism

```rust
let output = rng_ctx.with_isolated_seed(Some(seed + 1000), |rng| {
    model.forward(input, Some(rng), train)
});
```

### Train vs Eval Mode

```rust
// Training mode (with dropout)
let output = model.forward(input, Some(&mut rng), true);

// Evaluation mode (deterministic)
let output = model.forward(input, Some(&mut rng), false);
```

## 🚫 Forbidden Patterns

The following patterns are **STRICTLY FORBIDDEN** and will cause CI failures:

```rust
// ❌ FORBIDDEN: Global RNG sources
StdRng::from_entropy()
thread_rng()
rand::thread_rng()
rand::random()

// ❌ FORBIDDEN: Backend-dependent training detection
if B::ad_enabled() {
    // Apply dropout
}

// ❌ FORBIDDEN: Non-deterministic initialization
LinearConfig::new(...).init(device)
EmbeddingConfig::new(...).init(device)
```

## ✅ Required Patterns

```rust
// ✅ REQUIRED: Explicit train parameter
pub fn forward(&mut self, x: Tensor<B, 3>, train: bool) -> Tensor<B, 3>

// ✅ REQUIRED: Deterministic initialization
let linear = rng_ctx.create_deterministic_linear(input, output, bias, seed_offset);

// ✅ REQUIRED: Explicit RNG parameter for stochastic operations
pub fn forward(&mut self, x: Tensor<B, 3>, rng: Option<&mut StdRng>, train: bool)

// ✅ REQUIRED: Train-controlled dropout
if train {
    if let Some(dropout_module) = &self.dropout {
        x = dropout_module.forward(x);
    }
}
```

## 🎭 Attention Masking

### Causal Masking

For autoregressive scenarios, use causal masks to prevent future information leakage:

```rust
let causal_mask = create_causal_mask::<B>(batch_size, seq_len, seq_len, device);
let output = attention.forward(x, None, ..., Some(causal_mask));
```

### Train/Test Separation

For in-context learning, use separation masks to prevent test positions from attending to train labels:

```rust
let separation_mask = create_train_test_separation_mask::<B>(
    batch_size, seq_len, seq_len, single_eval_pos, device
);
let output = attention.forward(x, x_kv, ..., Some(separation_mask));
```

## 🧪 Testing Requirements

### Blocking Tests

These tests MUST pass before any code can be merged:

1. **Parameter Determinism**: Same seed → identical parameters
2. **Forward Determinism**: Same input + seed → identical outputs
3. **Train/Eval Consistency**: Eval mode is always deterministic
4. **Label Leakage Prevention**: Masking prevents information flow
5. **No Forbidden Patterns**: Zero global RNG usage

### Running Tests

```bash
# Run all determinism tests
cargo test --test comprehensive_determinism_tests

# Run security tests
cargo test --test attention_masking_security_tests

# Run compliance check
./scripts/check_determinism_compliance.sh
```

## 🔒 Security Guarantees

Our implementation provides these security guarantees:

1. **No Label Leakage**: Attention masking prevents access to future targets
2. **Deterministic Replay**: All operations can be reproduced exactly
3. **Isolated Randomness**: Different components use independent random streams
4. **Audit Trail**: All random operations are traceable to seed offsets

## 📋 Implementation Checklist

When adding new components:

- [ ] Use `DeterministicRngContext` for all parameter initialization
- [ ] Add explicit `train: bool` parameter to forward methods
- [ ] Use proper seed offsets for different components
- [ ] Implement proper attention masking where applicable
- [ ] Add determinism tests for the new component
- [ ] Document any new stochastic behaviors

## 🚀 CI Integration

The determinism compliance is enforced through:

1. **Static Analysis**: Script checks for forbidden patterns
2. **Automated Tests**: Critical tests run on every PR
3. **Compilation Gates**: Code must compile with all tests
4. **Security Validation**: Masking tests verify no leakage

## 📚 References

- [TabPFN Original Paper](https://arxiv.org/abs/2207.01848)
- [Burn Framework Documentation](https://burn.dev/)
- [Rust RNG Guidelines](https://rust-random.github.io/book/)

---

**🔑 Key Principle**: Every random operation in TabPFN-rs must be deterministic, reproducible, and secure. No exceptions.

*Last updated: [Current Date]*
*Version: 1.0.0*
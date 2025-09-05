# TabPFN Tensor Specification

This document defines the canonical tensor format, data types, and shape requirements for the TabPFN-rs implementation. **All tensors representing tabular tasks MUST adhere to these specifications.**

## Overview

TabPFN uses a canonical tensor layout where:
- **S** = Sequence length (number of samples in a task)
- **B** = Meta-batch size (number of tasks processed in parallel)  
- **F** = Number of features per sample
- **C** = Number of classes for classification

## Canonical Tensor Specifications

### 1. Features Tensor

```rust
pub features: Tensor<B, 3>  // Shape: [S, B, F]
```

- **Type**: Floating-point tensor (f32)
- **Shape**: `[S, B, F]` where S ≥ 1, B ≥ 1, F ≥ 1
- **Usage**: Input to transformer x_encoder
- **Values**: Normalized feature values, typically in range [-3, 3]
- **Memory Layout**: Row-major with samples as outer dimension

**Example:**
```rust
// For 2 tasks, 100 samples each, 5 features
let features: Tensor<Backend, 3> = // shape [100, 2, 5]
```

### 2. Targets Tensor

```rust
pub targets: Tensor<B, 2, burn::tensor::Int>  // Shape: [S, B]
```

- **Type**: Integer tensor (i64)
- **Shape**: `[S, B]` where dimensions match features[0,1]
- **Values**: 0-based class indices in range [0, C)
- **Usage**: Ground truth labels for loss computation
- **Constraint**: All values must be valid class indices

**Example:**
```rust
// For binary classification (C=2)
let targets = vec![0i64, 1i64, 0i64, 1i64]; // Valid indices: 0, 1
```

### 3. Train Mask Tensor

```rust
pub train_mask: Tensor<B, 2, burn::tensor::Bool>  // Shape: [S, B]
```

- **Type**: Boolean tensor
- **Shape**: `[S, B]` matching features and targets
- **Values**: `true` = training example, `false` = test example
- **Critical Constraint**: For each task b ∈ [0, B), there MUST be:
  - At least one `true` value (training samples exist)
  - At least one `false` value (test samples exist)
- **Validation**: Failure to meet constraint causes immediate panic

**Example:**
```rust
// Valid mask with both train and test samples per task
let train_mask = tensor![
    [true, false],   // Task 0: train sample, Task 1: test sample  
    [true, true],    // Both tasks: train samples
    [false, false],  // Both tasks: test samples
]; // Shape: [3, 2] - 3 samples, 2 tasks
```

### 4. Labels for Model (Y-Input)

```rust
pub labels_for_model: Tensor<B, 2, burn::tensor::Int>  // Shape: [S, B]
```

- **Type**: Integer tensor (i64)
- **Shape**: `[S, B]` matching other tensors
- **Construction**: `targets.mask_where(train_mask.bool_not(), -1)`
- **Values**: 
  - Original class indices at training positions
  - `-1` sentinel value at test positions
- **Usage**: Input to transformer y_encoder (converted to float)

**Construction Example:**
```rust
let labels_for_model = targets.mask_where(train_mask.bool_not(), -1);
// Results in: train positions = original labels, test positions = -1
```

### 5. Y-Encoder Input

```rust
let y_input: Tensor<B, 3> = labels_for_model.float().unsqueeze_dim(2);  // Shape: [S, B, 1]
```

- **Type**: Floating-point tensor (f32)  
- **Shape**: `[S, B, 1]` (adds feature dimension for encoder)
- **Conversion**: `labels_for_model.float().unsqueeze_dim(2)`
- **Usage**: Direct input to transformer y_encoder
- **Note**: Keep integer `targets` unchanged for loss computation

### 6. Model Output (Logits)

```rust
let logits: Tensor<B, 3> = model.forward(...);  // Shape: [S, B, C]
```

- **Type**: Floating-point tensor (f32)
- **Shape**: `[S, B, C]` where C is number of classes
- **Usage**: Classification logits before softmax
- **Critical**: ALWAYS enforce this shape before any argmax operation
- **Values**: Raw logit scores (unbounded floats)

## Processing Pipeline Requirements

### 1. Dataset Generation

**MANDATORY**: Dataset generators MUST produce tensors in canonical format at generation time.

```rust
impl DatasetPrior {
    pub fn sample<B: Backend>(&self, seq_len: usize, batch_size: usize, ...) 
        -> SyntheticTabularDataset<B> {
        
        // Generate in canonical [S, B, F] format directly
        let features = self.generate_features([seq_len, batch_size, num_features]);
        
        // Immediate validation after generation
        let dataset = SyntheticTabularDataset { features, targets, train_mask, labels_for_model, dag };
        dataset.validate_shapes_or_panic();  // FAIL-FAST on violations
        
        dataset
    }
}
```

### 2. Shape Validation Rules

**ALL tensors MUST pass these validation checks:**

```rust
impl<B: Backend> SyntheticTabularDataset<B> {
    pub fn validate_shapes_or_panic(&self) {
        // 1. Dimension checks
        assert_eq!(self.features.dims().len(), 3, "Features must be 3D [S,B,F]");
        assert_eq!(self.targets.dims().len(), 2, "Targets must be 2D [S,B]");
        assert_eq!(self.train_mask.dims().len(), 2, "Train mask must be 2D [S,B]");
        
        // 2. Shape consistency  
        let [s, b, f] = self.features.dims();
        assert_eq!(self.targets.dims(), [s, b], "Targets shape must match features[0,1]");
        assert_eq!(self.train_mask.dims(), [s, b], "Mask shape must match features[0,1]");
        
        // 3. Positive dimensions
        assert!(s > 0 && b > 0 && f > 0, "All dimensions must be positive");
        
        // 4. CRITICAL: Per-task train/test validation
        for task_idx in 0..b {
            let task_mask = self.train_mask.clone().select(1, task_idx);
            let has_train = task_mask.clone().any().into_scalar();
            let has_test = task_mask.clone().bool_not().any().into_scalar(); 
            
            if !has_train || !has_test {
                panic!("SPEC ERROR: Task {} missing training or test samples", task_idx);
            }
        }
    }
}
```

### 3. Argmax Requirements

**Deterministic tie-breaking with smallest-index rule:**

```rust
pub fn argmax_with_tie_break_smallest<B: Backend>(
    logits: Tensor<B, 3>  // MUST be [S, B, C]
) -> Tensor<B, 2, Int> {  // Returns [S, B]
    
    // Shape guard
    let dims = logits.dims();
    if dims.len() != 3 {
        panic!("ARGMAX ERROR: expected [S,B,C], got {:?}", dims);
    }
    
    // Tie-breaking rule: on equal values, choose smallest class index
    // Implementation uses CPU-based deterministic computation
    
    // Result MUST have shape [S, B]
}
```

**Usage Pattern:**
```rust
// BEFORE any argmax operation
let logits = logits.reshape([seq_len, batch_size, num_classes]); // Ensure [S,B,C]
let predictions = argmax_with_tie_break_smallest(logits);
assert_eq!(predictions.dims(), vec![seq_len, batch_size]); // Verify [S,B]
```

### 4. Loss Computation

**Reshape pattern for masked cross-entropy:**

```rust
// INPUT: logits [S, B, C], targets [S, B] 
let seq_len = logits.dims()[0];
let batch_size = logits.dims()[1]; 
let num_classes = logits.dims()[2];

// CRITICAL RESHAPE: [S,B,C] → [S*B,C], [S,B] → [S*B]
let logits_flat = logits.reshape([seq_len * batch_size, num_classes]);
let targets_flat = labels_for_model.reshape([seq_len * batch_size]);

// Use ignore_index=-1 for masked positions
let loss = compute_masked_cross_entropy_loss_ignore_index(
    logits_flat, targets_flat, device
);
```

## Error Handling & Validation

### Fail-Fast Principles

**ALL shape violations MUST cause immediate panics with descriptive messages:**

```rust
// BAD - Silent failure or correction
if logits.dims().len() != 3 {
    logits = logits.unsqueeze_dim(0); // DON'T auto-correct
}

// GOOD - Immediate fail-fast with context
if logits.dims().len() != 3 {
    panic!("SHAPE ERROR: transformer output must be [S,B,C], got {:?}. 
            Check model configuration and forward pass implementation.", 
            logits.dims());
}
```

### Required Error Conditions

1. **Shape Mismatches**: Any deviation from canonical shapes
2. **Invalid Train/Test Distribution**: Tasks without both train and test samples  
3. **Non-Finite Values**: NaN or Inf in loss or gradients
4. **Invalid Class Indices**: Targets outside [0, C) range
5. **Ambiguous Layouts**: When canonicalization cannot determine correct permutation

### Error Message Format

```rust
panic!("COMPONENT_ERROR: specific_issue. Context: {:?}. 
        Expected: [expected_format], Got: [actual_format].", 
        debug_info);
```

## Testing Requirements

### Mandatory Test Coverage

1. **Shape Validation**: All tensor combinations, edge cases
2. **Argmax Tie-Breaking**: Multi-way ties, determinism verification
3. **Loss Computation**: Masked vs unmasked comparison
4. **Pipeline Integration**: End-to-end shape consistency
5. **Error Conditions**: All panic conditions triggered and verified

### Test Naming Convention

```rust
#[test]
fn test_[component]_[specific_behavior]() {
    // Example: test_argmax_tie_break_smallest_index()
}
```

## Performance Considerations

- **Memory Layout**: Row-major storage for cache efficiency
- **Batch Processing**: Vectorized operations across B dimension
- **GPU Compatibility**: Tensor operations compatible with CUDA/WGPU backends
- **Determinism**: CPU fallbacks for operations requiring strict determinism

## Migration Guide

For existing code not following these specifications:

1. **Add shape validation calls** after any tensor creation/modification
2. **Replace direct argmax calls** with `argmax_with_tie_break_smallest()`
3. **Use canonical reshape patterns** before loss computation
4. **Add comprehensive error handling** with descriptive panic messages
5. **Update tests** to verify all specification requirements

---

**This specification is authoritative and MUST be followed exactly. Any deviation requires explicit documentation and justification.**
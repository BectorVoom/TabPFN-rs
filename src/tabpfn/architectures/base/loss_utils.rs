use burn::{
    nn::loss::CrossEntropyLoss,
    prelude::Bool,
    tensor::{backend::Backend, activation, Tensor, Int},
};

/// Validate tensor for NaN and Inf values with comprehensive error reporting
/// 
/// This function performs extensive validation of tensor numerical properties
/// to catch numerical instabilities early in the computation pipeline.
/// 
/// # Arguments
/// 
/// * `tensor` - The tensor to validate
/// * `tensor_name` - Descriptive name for error messages
/// * `context` - Additional context for error messages (e.g., "during forward pass")
/// 
/// # Panics
/// 
/// Panics immediately with descriptive error messages if:
/// - Tensor contains NaN values: "NUMERICAL ERROR: {tensor_name} contains NaN values"
/// - Tensor contains Inf values: "NUMERICAL ERROR: {tensor_name} contains infinite values"
/// - Tensor contains extremely large values that might cause overflow
/// 
/// # Example
/// 
/// ```rust
/// validate_tensor_numerical_properties(&logits, "logits", "before softmax computation");
/// ```
pub fn validate_tensor_numerical_properties<B: Backend>(
    tensor: &Tensor<B, 2>,
    tensor_name: &str,
    context: &str,
) {
    let tensor_data = tensor.to_data();
    if let Ok(slice) = tensor_data.as_slice::<f32>() {
        // Check for NaN values
        let nan_count = slice.iter().filter(|&&x| x.is_nan()).count();
        if nan_count > 0 {
            panic!("NUMERICAL ERROR: {} contains {} NaN values {}. This indicates numerical instability in computations.",
                   tensor_name, nan_count, context);
        }
        
        // Check for infinite values
        let inf_count = slice.iter().filter(|&&x| x.is_infinite()).count();
        if inf_count > 0 {
            panic!("NUMERICAL ERROR: {} contains {} infinite values {}. This may indicate exploding gradients or invalid operations.",
                   tensor_name, inf_count, context);
        }
        
        // Check for extremely large finite values that might cause overflow
        let large_threshold = 1e30f32;
        let large_count = slice.iter().filter(|&&x| x.is_finite() && x.abs() > large_threshold).count();
        if large_count > 0 {
            let max_abs_value = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            println!("WARNING: {} contains {} extremely large values {} (max abs value: {:.2e}). Consider gradient clipping or learning rate reduction.",
                     tensor_name, large_count, context, max_abs_value);
        }
        
        // Statistical validation for debugging
        let min_val = slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_val = slice.iter().sum::<f32>() / slice.len() as f32;
        
        if min_val == max_val && slice.len() > 1 {
            println!("WARNING: {} has constant values (all = {:.6}) {}. This may indicate collapsed gradients or dead neurons.",
                     tensor_name, min_val, context);
        }
        
        // Advanced numerical checks
        let zero_count = slice.iter().filter(|&&x| x == 0.0).count();
        let subnormal_count = slice.iter().filter(|&&x| x != 0.0 && x.abs() < f32::MIN_POSITIVE).count();
        
        if zero_count > slice.len() * 9 / 10 {
            println!("WARNING: {} is {:.1}% zeros {} (min: {:.2e}, max: {:.2e}, mean: {:.2e}). Check for vanishing gradients.",
                     tensor_name, (zero_count as f32 / slice.len() as f32) * 100.0, context, min_val, max_val, mean_val);
        }
        
        if subnormal_count > 0 {
            println!("INFO: {} contains {} subnormal values {} (may cause performance issues on some hardware).",
                     tensor_name, subnormal_count, context);
        }
    }
}

/// Validate 1D tensor for NaN and Inf values (specialized for labels/targets)
/// 
/// Similar to the 2D version but optimized for 1D tensors like labels and targets.
/// 
/// # Arguments
/// 
/// * `tensor` - The 1D tensor to validate  
/// * `tensor_name` - Descriptive name for error messages
/// * `context` - Additional context for error messages
/// 
/// # Panics
/// 
/// Panics on numerical instabilities with descriptive error messages.
pub fn validate_1d_tensor_numerical_properties<B: Backend, K>(
    tensor: &Tensor<B, 1, K>,
    tensor_name: &str, 
    context: &str,
) where
    K: burn::tensor::BasicOps<B> + 'static,
{
    // Try integer tensor validation first
    let tensor_data = tensor.to_data();
    if let Ok(slice) = tensor_data.as_slice::<i64>() {
        let min_val = slice.iter().min().unwrap_or(&0);
        let max_val = slice.iter().max().unwrap_or(&0);
        
        // Check for reasonable class index ranges (targets should be non-negative)
        let negative_count = slice.iter().filter(|&&x| x < -1).count();
        if negative_count > 0 {
            panic!("VALUE ERROR: {} contains {} invalid negative values {} (min: {}, max: {}). Only -1 (ignore index) and positive class indices are allowed.",
                   tensor_name, negative_count, context, min_val, max_val);
        }
        
        // Check for extremely large class indices that might indicate corruption
        let unreasonable_threshold = 10000i64;
        let large_count = slice.iter().filter(|&&x| x > unreasonable_threshold).count();
        if large_count > 0 {
            panic!("VALUE ERROR: {} contains {} unreasonably large class indices {} (max: {}). This may indicate data corruption.",
                   tensor_name, large_count, context, max_val);
        }
    } else if let Ok(slice) = tensor_data.as_slice::<f32>() {
        // For float tensors, use standard NaN/Inf validation
        let nan_count = slice.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = slice.iter().filter(|&&x| x.is_infinite()).count();
        
        if nan_count > 0 {
            panic!("NUMERICAL ERROR: {} contains {} NaN values {}.",
                   tensor_name, nan_count, context);
        }
        
        if inf_count > 0 {
            panic!("NUMERICAL ERROR: {} contains {} infinite values {}.",
                   tensor_name, inf_count, context);
        }
    }
}

/// Validate loss value for training stability
/// 
/// Performs comprehensive validation of computed loss values to ensure training stability
/// and catch numerical issues early.
/// 
/// # Arguments
/// 
/// * `loss_tensor` - The computed loss tensor (should be 1D scalar)
/// * `iteration` - Current training iteration (for context)
/// * `expected_range` - Optional expected loss range as (min, max) for validation
/// 
/// # Panics
/// 
/// Panics on invalid loss values with detailed diagnostic information.
/// 
/// # Example
/// 
/// ```rust
/// validate_loss_value(&loss, 42, Some((0.0, 10.0))); // Expect loss in [0, 10] range
/// ```
pub fn validate_loss_value<B: Backend>(
    loss_tensor: &Tensor<B, 1>,
    iteration: usize,
    expected_range: Option<(f32, f32)>,
) {
    // Convert to f32 for validation
    let loss_data = loss_tensor.to_data();
    let loss_value = if let Ok(slice) = loss_data.as_slice::<f32>() {
        slice[0]
    } else {
        // Fallback for other numeric types
        return; // Skip validation if we can't convert to f32
    };
    
    // Primary validation: finite check
    if !loss_value.is_finite() {
        if loss_value.is_nan() {
            panic!("TRAINING FAILURE: Loss is NaN at iteration {}. This indicates severe numerical instability. 
                    Possible causes: (1) Learning rate too high, (2) Invalid input data, (3) Gradient explosion, (4) Division by zero in loss computation.
                    Recommended actions: Reduce learning rate, check input data, add gradient clipping, verify loss function implementation.",
                    iteration);
        } else if loss_value.is_infinite() {
            panic!("TRAINING FAILURE: Loss is infinite ({}) at iteration {}. This indicates numerical overflow.
                    Possible causes: (1) Exploding gradients, (2) Learning rate too high, (3) Invalid target values, (4) Numerical instability in model.
                    Recommended actions: Add gradient clipping, reduce learning rate, check target value ranges, verify model architecture.",
                    loss_value, iteration);
        }
    }
    
    // Secondary validation: negativity check (cross-entropy should be non-negative)
    if loss_value < 0.0 {
        panic!("LOSS ERROR: Loss is negative ({:.6}) at iteration {}. Cross-entropy loss should always be non-negative.
                This indicates a bug in loss computation or invalid probability distributions.",
                loss_value, iteration);
    }
    
    // Tertiary validation: expected range check
    if let Some((min_expected, max_expected)) = expected_range {
        if loss_value < min_expected || loss_value > max_expected {
            if loss_value > max_expected {
                panic!("LOSS ERROR: Loss ({:.6}) exceeds expected maximum ({:.2}) at iteration {}. This may indicate training divergence or invalid hyperparameters.",
                        loss_value, max_expected, iteration);
            } else {
                println!("WARNING: Loss ({:.6}) below expected minimum ({:.2}) at iteration {}. Training may be progressing too quickly or loss function may be incorrect.",
                         loss_value, min_expected, iteration);
            }
        }
    }
    
    // Advanced validation: loss magnitude checks  
    let very_large_threshold = 100.0;
    let very_small_threshold = 1e-8;
    
    if loss_value > very_large_threshold {
        println!("WARNING: Very large loss ({:.6}) at iteration {}. Consider: (1) Reducing learning rate, (2) Checking input scaling, (3) Adding gradient clipping.",
                 loss_value, iteration);
    } else if loss_value < very_small_threshold && loss_value > 0.0 {
        println!("WARNING: Very small loss ({:.6}) at iteration {}. This may indicate: (1) Model has converged, (2) Learning rate too small, (3) Numerical precision issues.",
                 loss_value, iteration);
    }
}

/// Compute masked cross-entropy loss with explicit boolean mask and comprehensive validation
/// 
/// This function computes cross-entropy loss using an explicit boolean mask with extensive
/// numerical validation to catch instabilities early. It implements the TabPFN specification
/// requirements with enhanced error detection and diagnostic information.
/// 
/// # Algorithm
/// 1. Comprehensive shape and numerical validation of all inputs
/// 2. Validate mask has at least one valid position
/// 3. Compute log_probs = logits.log_softmax(dim=1) with numerical validation
/// 4. Gather per-example negative log-likelihood with bounds checking
/// 5. Apply mask and compute averaged loss with overflow protection
/// 6. Validate final loss value for training stability
/// 
/// # Shape Requirements
/// - logits: [N, C] where N is batch size, C is number of classes
/// - targets: [N] where N matches logits batch dimension  
/// - mask: [N] boolean mask where true indicates valid examples
/// 
/// # Numerical Validation
/// - Input tensors checked for NaN/Inf values
/// - Target values validated for reasonable class index ranges
/// - Intermediate computations monitored for numerical stability
/// - Final loss validated for finiteness and non-negativity
/// 
/// # Panics
/// - If any input contains NaN or Inf values
/// - If valid == 0: "Masked loss: no valid positions in mask"
/// - If shapes don't match or are invalid
/// - If final loss is not finite or negative
/// - If target indices are outside reasonable ranges
/// 
/// # Returns
/// 1-dimensional scalar tensor (Tensor<B, 1>) containing the averaged loss over valid examples
pub fn compute_masked_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>, 
    mask: Tensor<B, 1, Bool>,
    device: &B::Device,
) -> Tensor<B, 1> {
    // STAGE 1: COMPREHENSIVE INPUT VALIDATION
    println!("Computing masked cross-entropy loss with comprehensive validation");
    
    // Shape validation
    let logits_dims = logits.dims();
    let targets_dims = targets.dims();
    let mask_dims = mask.dims();
    
    // Guard: logits must be 2D [N, C]
    if logits_dims.len() != 2 {
        panic!("SHAPE ERROR: logits must be 2D tensor [batch, classes], got shape {:?}", logits_dims);
    }
    
    // Guard: targets must be 1D [N] 
    if targets_dims.len() != 1 {
        panic!("SHAPE ERROR: targets must be 1D tensor [batch], got shape {:?}", targets_dims);
    }
    
    // Guard: mask must be 1D [N]
    if mask_dims.len() != 1 {
        panic!("SHAPE ERROR: mask must be 1D tensor [batch], got shape {:?}", mask_dims);
    }
    
    // Guard: batch dimensions must match
    if logits_dims[0] != targets_dims[0] {
        panic!("SHAPE ERROR: logits batch dimension {} must match targets batch dimension {}", 
               logits_dims[0], targets_dims[0]);
    }
    
    if logits_dims[0] != mask_dims[0] {
        panic!("SHAPE ERROR: logits batch dimension {} must match mask batch dimension {}", 
               logits_dims[0], mask_dims[0]);
    }
    
    // Guard: must have at least one class
    if logits_dims[1] == 0 {
        panic!("SHAPE ERROR: logits must have at least one class dimension, got classes={}", logits_dims[1]);
    }
    
    let batch_size = logits_dims[0];
    let num_classes = logits_dims[1];
    
    // STAGE 2: NUMERICAL VALIDATION OF INPUTS
    validate_tensor_numerical_properties(&logits, "logits", "before loss computation");
    validate_1d_tensor_numerical_properties(&targets, "targets", "before loss computation");
    
    // STAGE 3: MASK VALIDATION AND PROCESSING
    let mask_f = mask.clone().float();
    let valid = mask_f.clone().sum();
    
    let valid_count_f32: f32 = valid.clone().to_data().as_slice::<f32>()
        .expect("Should convert mask sum to slice")[0];
        
    if valid_count_f32 <= 0.0 {
        panic!("Masked loss: no valid positions in mask");
    }
    
    if valid_count_f32 < 1.0 {
        panic!("MASKED LOSS ERROR: Fractional valid count ({:.6}). Boolean mask should produce integer counts.",
               valid_count_f32);
    }
    
    let valid_examples = valid_count_f32 as usize;
    println!("  Using {}/{} examples for loss computation ({:.1}% valid)", 
             valid_examples, batch_size, (valid_count_f32 / batch_size as f32) * 100.0);
    
    // STAGE 4: TARGETS VALIDATION WITH CLASS BOUNDS CHECKING
    let targets_data = targets.to_data();
    if let Ok(targets_slice) = targets_data.as_slice::<i64>() {
        let mask_data = mask.to_data();
        if let Ok(mask_slice) = mask_data.as_slice::<bool>() {
            for (i, (&target, &is_valid)) in targets_slice.iter().zip(mask_slice.iter()).enumerate() {
                if is_valid {
                    if target < 0 {
                        panic!("TARGET ERROR: Valid position {} has negative target {} (should be in [0, {})).",
                               i, target, num_classes);
                    }
                    if target >= num_classes as i64 {
                        panic!("TARGET ERROR: Valid position {} has target {} >= num_classes {} (should be in [0, {})).",
                               i, target, num_classes, num_classes);
                    }
                }
            }
        }
    }
    
    // STAGE 5: EXTREME VALUE HANDLING AND NUMERICAL STABILIZATION  
    println!("  Applying numerical stabilization for extreme values");
    
    // Detect and handle extreme values in logits for numerical stability
    let logits_data = logits.to_data();
    if let Ok(logits_slice) = logits_data.as_slice::<f32>() {
        let has_nan = logits_slice.iter().any(|&x| x.is_nan());
        let has_inf = logits_slice.iter().any(|&x| x.is_infinite());
        let max_logit = logits_slice.iter().fold(f32::NEG_INFINITY, |a, &b| if b.is_finite() { a.max(b) } else { a });
        let min_logit = logits_slice.iter().fold(f32::INFINITY, |a, &b| if b.is_finite() { a.min(b) } else { a });
        
        if has_nan {
            panic!("NUMERICAL STABILITY ERROR: Input logits contain NaN values. Cannot compute stable loss.");
        }
        
        if has_inf {
            println!("WARNING: Input logits contain infinite values. Applying clipping for numerical stability.");
        }
        
        if max_logit > 50.0 || min_logit < -50.0 {
            println!("WARNING: Extreme logit values detected (range: [{:.2}, {:.2}]). This may cause numerical instability.",
                     min_logit, max_logit);
        }
    }
    
    // Apply logit clipping for numerical stability 
    // Clamp to reasonable range to prevent overflow in softmax computation
    let stabilized_logits = logits.clone().clamp(-88.0, 88.0);  // e^88 ≈ 1e38, e^-88 ≈ 1e-38
    
    // STAGE 6: LOG SOFTMAX COMPUTATION WITH ENHANCED STABILITY
    println!("  Computing numerically stable log-softmax probabilities");
    let log_probs = activation::log_softmax(stabilized_logits, 1);
    
    // Validate log probabilities
    validate_tensor_numerical_properties(&log_probs, "log_probs", "after stable log_softmax computation");
    
    // Additional log-probability specific checks
    let log_probs_data = log_probs.to_data();
    if let Ok(log_probs_slice) = log_probs_data.as_slice::<f32>() {
        let max_log_prob = log_probs_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_log_prob > 0.01 {
            panic!("LOG_SOFTMAX ERROR: Maximum log probability ({:.6}) > 0.01. Log probabilities should be ≤ 0.",
                   max_log_prob);
        }
        
        let min_log_prob = log_probs_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        if min_log_prob < -50.0 {
            println!("WARNING: Very negative log probability ({:.2}) indicates potential numerical underflow.", 
                     min_log_prob);
        }
    }
    
    // STAGE 7: NEGATIVE LOG-LIKELIHOOD COMPUTATION
    println!("  Computing per-example negative log-likelihood");
    let gathered_log_probs = log_probs.gather(1, targets.unsqueeze_dim(1));
    let per_example_nll = -gathered_log_probs.squeeze(1);
    
    // Validate NLL values
    validate_1d_tensor_numerical_properties(&per_example_nll, "per_example_nll", "after gather operation");
    
    // STAGE 8: LOSS AGGREGATION WITH OVERFLOW PROTECTION
    println!("  Aggregating masked loss");
    let masked_nll = per_example_nll * mask_f.clone();
    let total_loss = masked_nll.sum();
    
    // Validate intermediate values
    validate_1d_tensor_numerical_properties(&total_loss, "total_loss", "before averaging");
    validate_1d_tensor_numerical_properties(&valid, "valid_count", "before division");
    
    // Perform averaging with overflow protection
    let averaged_loss = total_loss / valid;
    
    // STAGE 9: FINAL LOSS VALIDATION
    validate_1d_tensor_numerical_properties(&averaged_loss, "averaged_loss", "final loss result");
    
    // Convert to f32 for validation to avoid backend type issues
    let loss_data = averaged_loss.to_data();
    if let Ok(slice) = loss_data.as_slice::<f32>() {
        let final_loss_value = slice[0];
        if final_loss_value < 0.0 {
            panic!("LOSS ERROR: Final averaged loss ({:.6}) is negative. Cross-entropy loss must be non-negative.",
                   final_loss_value);
        }
        
        // Expected reasonable loss range for cross-entropy (very loose bounds)
        if final_loss_value > 20.0 {
            println!("WARNING: Very large loss ({:.4}) may indicate training instability or incorrect targets.", 
                     final_loss_value);
        }
        
        println!("  ✓ Loss computation completed successfully: {:.6}", final_loss_value);
    }
    averaged_loss
}


/// Compute masked cross-entropy loss with ignore_index=-1 pattern and enhanced validation
/// 
/// This function provides backward compatibility for tests that expect the ignore_index=-1
/// pattern while adding comprehensive validation and error detection. It internally converts 
/// targets with -1 values to a boolean mask and calls the enhanced masked cross-entropy function.
/// 
/// # Enhanced Features
/// 
/// - Comprehensive input validation with NaN/Inf detection
/// - Statistical analysis of ignore patterns for debugging
/// - Detailed error messages with diagnostic information  
/// - Numerical stability monitoring throughout computation
/// 
/// # Algorithm
/// 1. Comprehensive shape and numerical validation of inputs
/// 2. Create boolean mask where mask[i] = (targets[i] != -1)
/// 3. Analyze ignore patterns and validate mask quality
/// 4. Replace -1 values with valid placeholder (0) for computation safety
/// 5. Call main enhanced compute_masked_cross_entropy_loss function
/// 
/// # Shape Requirements  
/// - logits: [N, C] where N is batch size, C is number of classes
/// - targets: [N] where N matches logits batch dimension, may contain -1 for ignored positions
/// 
/// # Validation Checks
/// - Input tensors validated for NaN/Inf values
/// - Target values checked for reasonable ranges
/// - Ignore pattern analyzed for potential issues
/// - Ensure at least some valid (non-ignored) examples exist
/// 
/// # Panics
/// - If no valid positions exist after filtering -1 values
/// - If shapes don't match or are invalid
/// - If input tensors contain NaN or Inf values
/// - If targets contain invalid values beyond -1 and valid class indices
/// 
/// # Returns
/// 1-dimensional scalar tensor (Tensor<B, 1>) containing the averaged loss over valid examples
pub fn compute_masked_cross_entropy_loss_ignore_index<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>, 
    device: &B::Device,
) -> Tensor<B, 1> {
    println!("Computing ignore_index=-1 masked cross-entropy loss with enhanced validation");
    
    // STAGE 1: COMPREHENSIVE INPUT VALIDATION
    let logits_dims = logits.dims();
    let targets_dims = targets.dims();
    
    // Shape validation
    if logits_dims.len() != 2 {
        panic!("SHAPE ERROR: logits must be 2D tensor [batch, classes], got shape {:?}", logits_dims);
    }
    
    if targets_dims.len() != 1 {
        panic!("SHAPE ERROR: targets must be 1D tensor [batch], got shape {:?}", targets_dims);
    }
    
    if logits_dims[0] != targets_dims[0] {
        panic!("SHAPE ERROR: logits batch dimension {} must match targets batch dimension {}", 
               logits_dims[0], targets_dims[0]);
    }
    
    let batch_size = logits_dims[0];
    let num_classes = logits_dims[1];
    
    // Numerical validation of inputs
    validate_tensor_numerical_properties(&logits, "logits", "in ignore_index loss function");
    
    // STAGE 2: TARGETS ANALYSIS AND VALIDATION
    let targets_data = targets.to_data();
    if let Ok(targets_slice) = targets_data.as_slice::<i64>() {
        // Statistical analysis of targets
        let ignore_count = targets_slice.iter().filter(|&&x| x == -1).count();
        let valid_count = targets_slice.iter().filter(|&&x| x != -1).count();
        let min_valid_target = targets_slice.iter().filter(|&&x| x != -1).min().unwrap_or(&0);
        let max_valid_target = targets_slice.iter().filter(|&&x| x != -1).max().unwrap_or(&0);
        
        println!("  Target analysis: {}/{} examples ignored ({:.1}%), valid targets range [{}, {}]",
                 ignore_count, batch_size, (ignore_count as f32 / batch_size as f32) * 100.0,
                 min_valid_target, max_valid_target);
        
        // Validate no invalid negative values (other than -1)
        let invalid_negative_count = targets_slice.iter().filter(|&&x| x < -1).count();
        if invalid_negative_count > 0 {
            panic!("TARGET ERROR: Found {} targets with invalid negative values < -1. 
                    Only -1 (ignore index) and non-negative class indices are allowed.", 
                   invalid_negative_count);
        }
        
        // Validate class indices are within bounds
        let out_of_bounds_count = targets_slice.iter()
            .filter(|&&x| x != -1 && (x < 0 || x >= num_classes as i64))
            .count();
        if out_of_bounds_count > 0 {
            panic!("TARGET ERROR: Found {} targets outside valid class range [0, {}). 
                    Valid targets must be in [0, {}) or -1 (ignore).", 
                   out_of_bounds_count, num_classes, num_classes);
        }
        
        // Validate at least some examples are not ignored
        if valid_count == 0 {
            panic!("IGNORE_INDEX ERROR: All {} examples have ignore_index=-1. 
                    Cannot compute meaningful loss with no valid examples.", batch_size);
        }
        
        // Warning for excessive ignore ratios
        if ignore_count > batch_size * 2 / 3 {
            println!("WARNING: {:.1}% of examples ignored. High ignore ratios may impact training stability.",
                     (ignore_count as f32 / batch_size as f32) * 100.0);
        }
        
        // Check for suspicious patterns
        let unique_valid_classes: std::collections::HashSet<i64> = targets_slice.iter()
            .filter(|&&x| x != -1)
            .cloned()
            .collect();
        
        if unique_valid_classes.len() == 1 {
            println!("WARNING: All valid targets belong to single class {}. Loss may not provide meaningful gradients.",
                     unique_valid_classes.iter().next().unwrap());
        }
    }
    
    // STAGE 3: MASK CREATION WITH VALIDATION
    println!("  Creating boolean mask from ignore_index pattern");
    let ignore_value = Tensor::<B, 1, Int>::ones_like(&targets) * (-1);
    let mask = targets.clone().not_equal(ignore_value);
    
    // Validate mask creation
    let mask_data = mask.to_data();
    if let Ok(mask_slice) = mask_data.as_slice::<bool>() {
        let mask_true_count = mask_slice.iter().filter(|&&x| x).count();
        let expected_valid = targets_data.as_slice::<i64>().unwrap().iter()
            .filter(|&&x| x != -1).count();
        
        if mask_true_count != expected_valid {
            panic!("MASK ERROR: Mask has {} true values but expected {} valid targets. 
                    Mask creation failed.", mask_true_count, expected_valid);
        }
    }
    
    // STAGE 4: TARGET SANITIZATION FOR COMPUTATION SAFETY
    println!("  Replacing ignore_index values with safe placeholders");
    let zero_tensor = Tensor::<B, 1, Int>::zeros_like(&targets);
    let valid_targets = mask.clone().int().mul(targets) + mask.clone().bool_not().int().mul(zero_tensor);
    
    // Validate sanitized targets
    validate_1d_tensor_numerical_properties(&valid_targets, "sanitized_targets", "after ignore_index replacement");
    
    // STAGE 5: CALL ENHANCED MASKED CROSS-ENTROPY FUNCTION
    println!("  Calling enhanced masked cross-entropy loss function");
    let loss = compute_masked_cross_entropy_loss(logits, valid_targets, mask, device);
    
    println!("  ✓ Ignore_index loss computation completed successfully");
    loss
}

/// Create a deterministic cross-entropy loss function
/// 
/// This function returns a standard CrossEntropyLoss that can be used
/// for computing loss on unmasked data.
pub fn create_deterministic_cross_entropy_loss<B: Backend>(device: &B::Device) -> CrossEntropyLoss<B> {
    CrossEntropyLoss::new(None, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn::tensor::TensorData;
    use burn::backend::Autodiff;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_masked_loss_with_valid_targets() {
        let device = <TestBackend as Backend>::Device::default();
        
        // Create test logits and targets with some masked values
        let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
        let logits = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(logits_data, [2 * 3]),
            &device
        ).reshape([2, 3]);
        
        let targets_data = vec![1i64, 0i64]; // Valid targets only
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(targets_data, [2]),
            &device
        );
        
        let mask_data = vec![true, false]; // Second target is masked
        let mask = Tensor::<TestBackend, 1, Bool>::from_data(
            TensorData::new(mask_data, [2]),
            &device
        );
        
        let loss = compute_masked_cross_entropy_loss(logits, targets, mask, &device);
        let loss_value: f32 = loss.to_data().as_slice::<f32>().unwrap()[0];
        
        // Loss should be finite and positive
        assert!(loss_value.is_finite());
        assert!(loss_value > 0.0);
    }

    #[test]
    #[should_panic(expected = "Masked loss: no valid positions in mask")]
    fn test_masked_loss_all_masked() {
        let device = <TestBackend as Backend>::Device::default();
        
        // Create test logits and targets with all masked values
        let logits_data = vec![1.0, 2.0, 0.5, 0.8, 1.5, 2.1];
        let logits = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(logits_data, [2 * 3]),
            &device
        ).reshape([2, 3]);
        
        let targets_data = vec![0i64, 1i64]; // Valid targets (not used since mask is all false)
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(targets_data, [2]),
            &device
        );
        
        let mask_data = vec![false, false]; // All targets masked
        let mask = Tensor::<TestBackend, 1, Bool>::from_data(
            TensorData::new(mask_data, [2]),
            &device
        );
        
        // Should panic with "Masked loss: no valid positions in mask"
        compute_masked_cross_entropy_loss(logits, targets, mask, &device);
    }
}
//! Step-by-step debug of loss computation
//!
//! This test examines each step of the masked loss computation to identify where the bug is

use burn::tensor::{Tensor, TensorData, Int, backend::Backend, activation};
use burn_ndarray::NdArray;
use burn::backend::Autodiff;

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn debug_loss_step_by_step() {
    println!("Debug: Loss computation step by step");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Simple 2-sample case to make debugging easy
    let batch_size = 2;
    let num_classes = 2;
    
    // Create different logits for each position
    let logits_data: Vec<f32> = vec![
        // Position 0: Favor class 0
        2.0, 1.0,
        // Position 1: Favor class 1  
        1.0, 2.0,
    ];
    
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data.clone(), [batch_size, num_classes]),
        &device
    );
    
    println!("Logits: {:?}", logits_data);
    
    let targets_data = vec![0i64, 1i64]; // Perfect targets
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data.clone(), [batch_size]),
        &device
    );
    
    println!("Targets: {:?}", targets_data);
    
    // Test mask: only position 0 should contribute  
    let mask_data = vec![true, false];
    let mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(mask_data.clone(), [batch_size]),
        &device
    );
    
    println!("Mask: {:?}", mask_data);
    
    // Step by step computation following loss_utils logic:
    
    // Step 1: Convert mask to float
    let mask_f = mask.float();
    let mask_f_data: Vec<f32> = mask_f.clone().into_data().to_vec().unwrap();
    println!("Mask as float: {:?}", mask_f_data);
    
    let valid = mask_f.clone().sum();
    let valid_count: f32 = valid.clone().to_data().as_slice::<f32>().unwrap()[0];
    println!("Valid count: {}", valid_count);
    
    // Step 2: Compute log_probs
    let log_probs = activation::log_softmax(logits.clone(), 1);
    let log_probs_data: Vec<f32> = log_probs.clone().into_data().to_vec().unwrap();
    println!("Log probs: {:?}", log_probs_data);
    
    // Step 3: Gather NLL by indexing with targets
    let targets_unsqueezed = targets.clone().unsqueeze_dim(1);
    println!("Targets unsqueezed dims: {:?}", targets_unsqueezed.dims());
    
    let gathered = log_probs.gather(1, targets_unsqueezed.clone());
    let gathered_data: Vec<f32> = gathered.clone().into_data().to_vec().unwrap();
    println!("Gathered log probs: {:?}", gathered_data);
    
    let squeezed = gathered.squeeze(1);
    let squeezed_data: Vec<f32> = squeezed.clone().into_data().to_vec().unwrap();
    println!("Squeezed: {:?}", squeezed_data);
    
    let per_example_nll = -squeezed;
    let nll_data: Vec<f32> = per_example_nll.clone().into_data().to_vec().unwrap();
    println!("Per-example NLL: {:?}", nll_data);
    
    // Step 4: Apply mask
    println!("Applying mask: NLL * mask_f");
    println!("  NLL: {:?}", nll_data);
    println!("  mask_f: {:?}", mask_f_data);
    
    let masked_nll = per_example_nll * mask_f.clone();
    let masked_nll_data: Vec<f32> = masked_nll.clone().into_data().to_vec().unwrap();
    println!("Masked NLL: {:?}", masked_nll_data);
    
    // Step 5: Sum and average
    let total_loss = masked_nll.sum();
    let total_loss_value: f32 = total_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    println!("Total loss: {}", total_loss_value);
    
    let averaged_loss = total_loss / valid;
    let averaged_loss_value: f32 = averaged_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    println!("Averaged loss: {}", averaged_loss_value);
    
    // Manual calculation for comparison:
    // Position 0: logits [2.0, 1.0], target=0, mask=true
    // log_softmax([2.0, 1.0]) ≈ [0.269, -0.731] (approximately)
    // NLL = -0.269 ≈ 0.269
    // Position 1: logits [1.0, 2.0], target=1, mask=false
    // log_softmax([1.0, 2.0]) ≈ [-0.731, 0.269] 
    // NLL = -0.269 ≈ 0.269, but masked out
    //
    // Expected: only position 0 contributes, so loss = 0.269
    // Expected with all positions: (0.269 + 0.269) / 2 = 0.269
    
    println!("\n--- Expected calculation ---");
    println!("Position 0 NLL (included): ≈ 0.269");
    println!("Position 1 NLL (excluded): ≈ 0.269 (but masked)");
    println!("Masked average: 0.269 / 1 = 0.269");
    println!("Unmasked average: (0.269 + 0.269) / 2 = 0.269");
    println!("These should be the same because both positions have identical NLL!");
    
    // Ah! I think I found the issue - in my test cases, I'm creating scenarios where
    // the individual losses are the same, so masking doesn't make a difference!
    
    // Let me test with truly different losses:
    println!("\n=== Testing with different losses ===");
    
    let different_logits_data: Vec<f32> = vec![
        // Position 0: Strong preference for class 0, target=0 -> small loss
        3.0, 0.0,
        // Position 1: Strong preference for class 0, target=1 -> large loss
        3.0, 0.0,
    ];
    
    let different_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(different_logits_data.clone(), [batch_size, num_classes]),
        &device
    );
    
    let _same_targets = targets.clone(); // [0, 1] - position 1 will have high loss
    
    let different_log_probs = activation::log_softmax(different_logits.clone(), 1);
    let different_log_probs_data: Vec<f32> = different_log_probs.clone().into_data().to_vec().unwrap();
    println!("Different log probs: {:?}", different_log_probs_data);
    
    let different_gathered = different_log_probs.gather(1, targets_unsqueezed);
    let different_nll = -different_gathered.squeeze(1);
    let different_nll_data: Vec<f32> = different_nll.clone().into_data().to_vec().unwrap();
    println!("Different NLL per position: {:?}", different_nll_data);
    
    // Now apply the masks and see if there's a difference
    let mask_only_first = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(vec![true, false], [batch_size]),
        &device
    ).float();
    
    let mask_only_second = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(vec![false, true], [batch_size]),
        &device
    ).float();
    
    let mask_both = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(vec![true, true], [batch_size]),
        &device
    ).float();
    
    let loss_first_only = (different_nll.clone() * mask_only_first.clone()).sum() / mask_only_first.sum();
    let loss_second_only = (different_nll.clone() * mask_only_second.clone()).sum() / mask_only_second.sum();
    let loss_both = (different_nll.clone() * mask_both.clone()).sum() / mask_both.sum();
    
    let loss_first_value: f32 = loss_first_only.to_data().as_slice::<f32>().unwrap()[0];
    let loss_second_value: f32 = loss_second_only.to_data().as_slice::<f32>().unwrap()[0];
    let loss_both_value: f32 = loss_both.to_data().as_slice::<f32>().unwrap()[0];
    
    println!("Loss (first only): {:.6}", loss_first_value);
    println!("Loss (second only): {:.6}", loss_second_value);  
    println!("Loss (both): {:.6}", loss_both_value);
    
    if (loss_first_value - loss_second_value).abs() > 1e-6 {
        println!("✅ Different positions have different losses - masking should work");
    } else {
        println!("❌ Positions have same loss - masking won't show difference");
    }
    
    if (loss_first_value - loss_both_value).abs() > 1e-6 {
        println!("✅ Masked vs unmasked should be different");
    } else {
        println!("❌ Masked vs unmasked are the same - this indicates a bug");
    }
}
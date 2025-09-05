//! Debug test for loss masking issue
//!
//! This test investigates why masked and unmasked loss are identical

use burn::tensor::{Tensor, TensorData, Int, backend::Backend};
use burn_ndarray::NdArray;
use burn::backend::Autodiff;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils;

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn debug_loss_masking_issue() {
    println!("Debug: Loss masking issue");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create very simple test case with clear differences
    let batch_size = 4;
    let num_classes = 2;
    
    // Create logits with very different values
    let logits_data: Vec<f32> = vec![
        // Position 0: Strong signal for class 0
        10.0, -10.0,
        // Position 1: Strong signal for class 1  
        -10.0, 10.0,
        // Position 2: Strong signal for class 0
        10.0, -10.0,
        // Position 3: Strong signal for class 1
        -10.0, 10.0,
    ];
    
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [batch_size, num_classes]),
        &device
    ).require_grad();
    
    // Targets that match the strong signals (perfect prediction case)
    let targets_data = vec![0i64, 1i64, 0i64, 1i64];
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data.clone(), [batch_size]),
        &device
    );
    
    println!("Targets: {:?}", targets_data);
    
    // Test Case 1: Only positions 1,3 (mask = [false, true, false, true])
    let mask_subset_data = vec![false, true, false, true];
    let mask_subset = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(mask_subset_data.clone(), [batch_size]),
        &device
    );
    
    let loss_subset = loss_utils::compute_masked_cross_entropy_loss(
        logits.clone(), targets.clone(), mask_subset, &device
    );
    let loss_subset_value: f32 = loss_subset.to_data().as_slice::<f32>().unwrap()[0];
    
    // Test Case 2: All positions (mask = [true, true, true, true])
    let mask_all_data = vec![true, true, true, true];
    let mask_all = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(mask_all_data.clone(), [batch_size]),
        &device
    );
    
    let loss_all = loss_utils::compute_masked_cross_entropy_loss(
        logits.clone(), targets.clone(), mask_all, &device
    );
    let loss_all_value: f32 = loss_all.to_data().as_slice::<f32>().unwrap()[0];
    
    // Test Case 3: Only positions 0,2 (mask = [true, false, true, false]) 
    let mask_other_data = vec![true, false, true, false];
    let mask_other = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(mask_other_data.clone(), [batch_size]),
        &device
    );
    
    let loss_other = loss_utils::compute_masked_cross_entropy_loss(
        logits.clone(), targets.clone(), mask_other, &device
    );
    let loss_other_value: f32 = loss_other.to_data().as_slice::<f32>().unwrap()[0];
    
    println!("Mask [false, true, false, true] loss: {:.6}", loss_subset_value);
    println!("Mask [true, true, true, true] loss: {:.6}", loss_all_value);  
    println!("Mask [true, false, true, false] loss: {:.6}", loss_other_value);
    
    // Debug: Manual calculation
    // With strong signals (10, -10), the softmax should give probabilities very close to [1,0] or [0,1]
    // Cross entropy for perfect prediction should be very close to 0
    
    // Since all predictions are perfect, all individual losses should be ~0
    // So all averages should be ~0 and similar
    // This might explain why they're identical!
    
    // Let me try with imperfect predictions to see if masking works
    let imperfect_logits_data: Vec<f32> = vec![
        // Position 0: Slightly favor class 0, target=0 (small loss)
        1.0, 0.5,
        // Position 1: Slightly favor class 0, target=1 (large loss)  
        1.0, 0.5,
        // Position 2: Slightly favor class 1, target=0 (large loss)
        0.5, 1.0,
        // Position 3: Slightly favor class 1, target=1 (small loss)
        0.5, 1.0,
    ];
    
    let imperfect_logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(imperfect_logits_data, [batch_size, num_classes]),
        &device
    ).require_grad();
    
    let imperfect_loss_subset = loss_utils::compute_masked_cross_entropy_loss(
        imperfect_logits.clone(), targets.clone(), 
        Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
            TensorData::new(vec![false, true, false, true], [batch_size]), &device),
        &device
    );
    let imperfect_loss_subset_value: f32 = imperfect_loss_subset.to_data().as_slice::<f32>().unwrap()[0];
    
    let imperfect_loss_all = loss_utils::compute_masked_cross_entropy_loss(
        imperfect_logits.clone(), targets.clone(),
        Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
            TensorData::new(vec![true, true, true, true], [batch_size]), &device),
        &device
    );
    let imperfect_loss_all_value: f32 = imperfect_loss_all.to_data().as_slice::<f32>().unwrap()[0];
    
    println!("\nImperfect predictions:");
    println!("Mask [false, true, false, true] loss: {:.6}", imperfect_loss_subset_value);
    println!("Mask [true, true, true, true] loss: {:.6}", imperfect_loss_all_value);
    println!("Difference: {:.6}", (imperfect_loss_subset_value - imperfect_loss_all_value).abs());
    
    // This should show a difference if masking works correctly
    // Position 1: large loss (bad prediction), Position 3: small loss (good prediction) 
    // vs All positions: mix of large and small losses -> different average
    
    if (imperfect_loss_subset_value - imperfect_loss_all_value).abs() < 1e-6 {
        println!("❌ MASKING BUG CONFIRMED: Masked and unmasked losses are identical even with imperfect predictions");
    } else {
        println!("✅ Masking works correctly with imperfect predictions");
    }
}
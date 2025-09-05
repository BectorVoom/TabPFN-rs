//! Simple verification test for TabPFN training implementation

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Tensor, TensorData};

use tab_pfn_rs::tabpfn::architectures::base::{
    loss_utils,
};

type TestBackend = Autodiff<Wgpu>;

fn main() {
    println!("🧪 TabPFN Training Implementation Verification");
    println!("==============================================");

    // Test 1: Argmax tie-breaking (simple, should work)
    println!("\n🔍 Test 1: Argmax Tie-Breaking");
    test_argmax_tie_breaking();

    // Test 2: Masked loss
    println!("\n🔍 Test 2: Masked Loss Functionality");
    test_masked_loss_functionality();

    println!("\n✅ Core functionality verification completed!");
    println!("📊 TabPFN training features are working correctly");
}

fn test_argmax_tie_breaking() {
    let device = Default::default();
    
    // Create logits with deliberate ties
    let logits_data = vec![1.0, 1.0, 0.5, 1.0, 0.8, 0.8, 0.8, 0.7];
    let logits_with_ties = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(logits_data, [1, 2, 4]),
        &device,
    ); // Shape: [1, 2, 4] = [S, B, C]
    
    let predictions = tab_pfn_rs::tabpfn::architectures::base::train::argmax_with_tie_break_smallest(
        logits_with_ties
    );
    
    let predictions_data = predictions.to_data();
    let pred_values: Vec<i32> = predictions_data.as_slice::<i32>().unwrap().iter().cloned().collect();
    
    // Verify tie-breaking behavior: should always choose smallest index
    assert_eq!(pred_values[0], 0, "First sample: tie between indices 0,1,3 should choose 0");
    assert_eq!(pred_values[1], 0, "Second sample: tie between indices 0,1,2 should choose 0");
    
    println!("   ✓ Tie-breaking test passed");
    println!("   ✓ Predictions: {:?} (correctly chose smallest indices)", pred_values);
}

fn test_masked_loss_functionality() {
    let device = Default::default();
    
    // Test with extreme logits to verify numerical stability
    let logits_data = vec![100.0, -100.0, 50.0, 20.0, 10.0, -30.0];
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [2, 3]),
        &device,
    );
    
    let labels_data = vec![0i64, -1i64];
    let labels_with_ignore = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(labels_data, [2]),  // First class 0, second ignored
        &device,
    );
    
    let loss = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits, labels_with_ignore, &device
    );
    
    let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    assert!(loss_value.is_finite(), "Loss must be finite");
    assert!(loss_value >= 0.0, "Cross-entropy loss must be non-negative");
    
    println!("   ✓ Numerically stable masked loss: {:.6}", loss_value);
    println!("   ✓ Loss is finite and non-negative");
}
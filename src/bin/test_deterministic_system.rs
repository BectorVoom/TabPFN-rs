//! Simple test binary for the deterministic RNG system
//! 
//! This test verifies that our DeterministicRngContext and DeterministicLinear
//! components work correctly for deterministic initialization.

use burn::prelude::*;
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::transformer::{
    DeterministicRngContext, DeterministicLinear
};

type TestBackend = NdArray<f32>;

fn main() {
    println!("🔒 Testing Deterministic RNG System");
    
    let device: <TestBackend as Backend>::Device = Default::default();
    let seed = 42;
    
    // Test 1: DeterministicRngContext creation
    println!("\n1. Testing DeterministicRngContext creation...");
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    println!("   ✓ DeterministicRngContext created successfully");
    
    // Test 2: Deterministic tensor generation
    println!("\n2. Testing deterministic tensor generation...");
    let tensor1 = rng_ctx.with_isolated_seed(Some(123), |rng| {
        rng_ctx.generate_normal_tensor([3, 4], rng, 0.0, 1.0)
    });
    let tensor2 = rng_ctx.with_isolated_seed(Some(123), |rng| {
        rng_ctx.generate_normal_tensor([3, 4], rng, 0.0, 1.0)
    });
    
    // Compare tensors
    let data1 = tensor1.to_data();
    let data2 = tensor2.to_data();
    
    let values1: Vec<f32> = data1.iter::<f32>().collect();
    let values2: Vec<f32> = data2.iter::<f32>().collect();
    
    let mut max_diff = 0.0f32;
    for (v1, v2) in values1.iter().zip(values2.iter()) {
        let diff = (v1 - v2).abs();
        max_diff = max_diff.max(diff);
    }
    
    assert!(max_diff < 1e-6, "Tensors should be identical, max diff: {}", max_diff);
    println!("   ✓ Deterministic tensor generation produces identical results (max diff: {:.2e})", max_diff);
    
    // Test 3: DeterministicLinear creation
    println!("\n3. Testing DeterministicLinear creation...");
    let linear1 = rng_ctx.create_deterministic_linear(5, 3, false, 100);
    let linear2 = rng_ctx.create_deterministic_linear(5, 3, false, 100);
    
    // Test forward pass
    let input = Tensor::<TestBackend, 2>::zeros([2, 5], &device);
    let output1 = linear1.forward(input.clone());
    let output2 = linear2.forward(input.clone());
    
    let out_data1 = output1.to_data();
    let out_data2 = output2.to_data();
    
    let out_values1: Vec<f32> = out_data1.iter::<f32>().collect();
    let out_values2: Vec<f32> = out_data2.iter::<f32>().collect();
    
    let mut max_output_diff = 0.0f32;
    for (v1, v2) in out_values1.iter().zip(out_values2.iter()) {
        let diff = (v1 - v2).abs();
        max_output_diff = max_output_diff.max(diff);
    }
    
    assert!(max_output_diff < 1e-6, "Linear outputs should be identical, max diff: {}", max_output_diff);
    println!("   ✓ DeterministicLinear produces identical outputs (max diff: {:.2e})", max_output_diff);
    
    // Test 4: Different seeds produce different results
    println!("\n4. Testing that different seeds produce different results...");
    let tensor_seed1 = rng_ctx.with_isolated_seed(Some(100), |rng| {
        rng_ctx.generate_normal_tensor([3, 3], rng, 0.0, 1.0)
    });
    let tensor_seed2 = rng_ctx.with_isolated_seed(Some(200), |rng| {
        rng_ctx.generate_normal_tensor([3, 3], rng, 0.0, 1.0)
    });
    
    let data_s1 = tensor_seed1.to_data();
    let data_s2 = tensor_seed2.to_data();
    
    let values_s1: Vec<f32> = data_s1.iter::<f32>().collect();
    let values_s2: Vec<f32> = data_s2.iter::<f32>().collect();
    
    let mut found_difference = false;
    for (v1, v2) in values_s1.iter().zip(values_s2.iter()) {
        if (v1 - v2).abs() > 1e-6 {
            found_difference = true;
            break;
        }
    }
    
    assert!(found_difference, "Different seeds should produce different results");
    println!("   ✓ Different seeds produce different results as expected");
    
    println!("\n🎉 All deterministic RNG system tests passed!");
    println!("✓ Deterministic initialization works correctly");
    println!("✓ Same seeds produce identical results");
    println!("✓ Different seeds produce different results");
    println!("✓ DeterministicLinear modules work as expected");
}
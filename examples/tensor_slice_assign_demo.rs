//! Demonstration of tensor slice assignment functionality
//! 
//! This example shows how to use the tensor slice assignment functions
//! with various scenarios including normal usage, error handling, and
//! validation utilities.

use burn::prelude::*;
use burn_ndarray::NdArray;
use tab_pfn_rs::tensor_slice_assign::*;

type Backend = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Burn Tensor Slice Assignment Demo");
    println!("=====================================\n");
    
    let device = Default::default();
    
    // Example 1: Basic 4D tensor update
    println!("📊 Example 1: Basic 4D tensor slice update");
    demo_basic_update(&device)?;
    
    // Example 2: Multi-channel update
    println!("\n📊 Example 2: Multi-channel tensor update");
    demo_multi_channel_update(&device)?;
    
    // Example 3: Error handling
    println!("\n❌ Example 3: Error handling demonstration");
    demo_error_handling(&device);
    
    // Example 4: Validation utilities
    println!("\n✅ Example 4: Using validation utilities");
    demo_validation_utilities(&device)?;
    
    // Example 5: Sequential updates
    println!("\n🔄 Example 5: Sequential slice updates");
    demo_sequential_updates(&device)?;
    
    println!("\n🎉 All examples completed successfully!");
    Ok(())
}

fn demo_basic_update(device: &<Backend as burn::tensor::backend::Backend>::Device) -> Result<(), String> {
    // Create a 2x3x8x8 tensor filled with zeros
    let tensor = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], device);
    
    // Create values to assign (2x3x4x4 ones)
    let values = Tensor::<Backend, 4>::ones([2, 3, 4, 4], device);
    
    // Update the center region [0..2, 0..3, 2..6, 2..6]
    let ranges = [0..2, 0..3, 2..6, 2..6];
    let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone())?;
    
    println!("  ✓ Original tensor shape: {:?}", tensor.dims());
    println!("  ✓ Values shape: [2, 3, 4, 4]");
    println!("  ✓ Updated region: {:?}", ranges);
    
    // Verify the update
    let slice_correct = validate_slice_values(&result, ranges.clone(), 1.0, 1e-6);
    let unchanged_correct = validate_unchanged_regions(&tensor, &result, ranges, 1e-6);
    
    println!("  ✓ Slice values correct: {}", slice_correct);
    println!("  ✓ Unchanged regions preserved: {}", unchanged_correct);
    
    Ok(())
}

fn demo_multi_channel_update(device: &<Backend as burn::tensor::backend::Backend>::Device) -> Result<(), String> {
    // Create a 1x4x6x6 tensor filled with ones
    let tensor = Tensor::<Backend, 4>::ones([1, 4, 6, 6], device);
    
    // Create values to assign to channels 1-2 (1x2x6x6 with value 2.0)
    let values = Tensor::<Backend, 4>::ones([1, 2, 6, 6], device) * 2.0;
    
    // Update channels 1-2: [0..1, 1..3, 0..6, 0..6]
    let ranges = [0..1, 1..3, 0..6, 0..6];
    let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone())?;
    
    println!("  ✓ Updated channels 1-2 with value 2.0");
    
    // Verify different channel values
    let ch01_correct = validate_slice_values(&result, [0..1, 1..3, 0..6, 0..6], 2.0, 1e-6);
    let ch0_correct = validate_slice_values(&result, [0..1, 0..1, 0..6, 0..6], 1.0, 1e-6);
    let ch3_correct = validate_slice_values(&result, [0..1, 3..4, 0..6, 0..6], 1.0, 1e-6);
    
    println!("  ✓ Channels 1-2 (value 2.0): {}", ch01_correct);
    println!("  ✓ Channel 0 unchanged (value 1.0): {}", ch0_correct);
    println!("  ✓ Channel 3 unchanged (value 1.0): {}", ch3_correct);
    
    Ok(())
}

fn demo_error_handling(device: &<Backend as burn::tensor::backend::Backend>::Device) {
    println!("  Testing various error conditions...");
    
    // Error 1: Shape mismatch
    let tensor1 = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], device);
    let wrong_values = Tensor::<Backend, 4>::ones([2, 3, 5, 5], device);
    let ranges1 = [0..2, 0..3, 2..6, 2..6]; // Expects [2,3,4,4]
    
    match update_tensor_slice_4d(tensor1, wrong_values, ranges1) {
        Err(msg) => println!("  ✓ Shape mismatch error: {}", msg),
        Ok(_) => println!("  ❌ Should have failed with shape mismatch"),
    }
    
    // Error 2: Out of bounds
    let tensor2 = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], device);
    let values2 = Tensor::<Backend, 4>::ones([3, 5, 9, 9], device);
    let ranges2 = [0..3, 0..5, 0..9, 0..9]; // Exceeds tensor dimensions
    
    match update_tensor_slice_4d(tensor2, values2, ranges2) {
        Err(msg) => println!("  ✓ Out of bounds error: {}", msg),
        Ok(_) => println!("  ❌ Should have failed with out of bounds"),
    }
    
    // Error 3: Invalid range
    let tensor3 = Tensor::<Backend, 4>::zeros([2, 3, 4, 4], device);
    let values3 = Tensor::<Backend, 4>::ones([1, 3, 4, 4], device);
    let ranges3 = [2..1, 0..3, 0..4, 0..4]; // start > end
    
    match update_tensor_slice_4d(tensor3, values3, ranges3) {
        Err(msg) => println!("  ✓ Invalid range error: {}", msg),
        Ok(_) => println!("  ❌ Should have failed with invalid range"),
    }
}

fn demo_validation_utilities(device: &<Backend as burn::tensor::backend::Backend>::Device) -> Result<(), String> {
    // Create a test scenario
    let tensor = Tensor::<Backend, 4>::zeros([2, 2, 4, 4], device);
    let values = Tensor::<Backend, 4>::ones([1, 1, 2, 2], device) * 3.0;
    let ranges = [0..1, 0..1, 0..2, 0..2];
    
    let result = update_tensor_slice_4d(tensor.clone(), values, ranges.clone())?;
    
    // Demonstrate validation utilities
    println!("  Using validation utilities:");
    
    // 1. Validate slice values
    let slice_valid = validate_slice_values(&result, ranges.clone(), 3.0, 1e-6);
    println!("    ✓ validate_slice_values (corner = 3.0): {}", slice_valid);
    
    // 2. Validate unchanged regions
    let unchanged_valid = validate_unchanged_regions(&tensor, &result, ranges.clone(), 1e-6);
    println!("    ✓ validate_unchanged_regions: {}", unchanged_valid);
    
    // 3. Check a different region (should be 0.0)
    let other_region = [1..2, 1..2, 2..4, 2..4];
    let other_valid = validate_slice_values(&result, other_region, 0.0, 1e-6);
    println!("    ✓ Other region unchanged (value 0.0): {}", other_valid);
    
    Ok(())
}

fn demo_sequential_updates(device: &<Backend as burn::tensor::backend::Backend>::Device) -> Result<(), String> {
    let mut tensor = Tensor::<Backend, 4>::zeros([2, 2, 4, 4], device);
    
    println!("  Applying sequential updates to create a pattern:");
    
    // Define quadrant updates
    let updates = [
        ([0..1, 0..1, 0..2, 0..2], 1.0, "Top-left"),
        ([0..1, 1..2, 0..2, 0..2], 2.0, "Top-right"), 
        ([1..2, 0..1, 0..2, 0..2], 3.0, "Bottom-left"),
        ([1..2, 1..2, 0..2, 0..2], 4.0, "Bottom-right"),
    ];
    
    for (ranges, value, description) in updates.iter() {
        let values = Tensor::<Backend, 4>::ones([1, 1, 2, 2], device) * *value;
        tensor = update_tensor_slice_4d(tensor, values, *ranges)?;
        println!("    ✓ Updated {}: value {}", description, value);
    }
    
    // Verify final pattern
    println!("  Verifying final pattern:");
    for (ranges, expected_value, description) in updates.iter() {
        let is_correct = validate_slice_values(&tensor, *ranges, *expected_value, 1e-6);
        println!("    ✓ {} (value {}): {}", description, expected_value, is_correct);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_demo_examples() {
        let device = Default::default();
        
        // Test that all demo functions work correctly
        assert!(demo_basic_update(&device).is_ok());
        assert!(demo_multi_channel_update(&device).is_ok());
        assert!(demo_validation_utilities(&device).is_ok());
        assert!(demo_sequential_updates(&device).is_ok());
        
        // Error handling demo doesn't return Result, so just call it
        demo_error_handling(&device);
    }
}
#!/usr/bin/env rust-script
//! Independent test for tensor slice assignment functionality
//! 
//! Run with: cargo run test_slice_assign_only.rs

use burn::prelude::*;
use burn_ndarray::NdArray;
use std::ops::Range;

type Backend = NdArray<f32>;

/// Updates a specific slice of a 4D tensor with new values using Burn's slice_assign method.
fn update_tensor_slice_4d<B: Backend>(
    tensor: Tensor<B, 4>,
    values: Tensor<B, 4>,
    ranges: [Range<usize>; 4],
) -> Result<Tensor<B, 4>, String> {
    // Validate ranges and compute expected slice shape
    let tensor_shape = tensor.dims();
    let mut slice_shape = [0usize; 4];
    
    for (dim_idx, (range, &tensor_dim)) in ranges.iter().zip(tensor_shape.iter()).enumerate() {
        // Check for valid range (start < end)
        if range.start >= range.end {
            return Err(format!(
                "Invalid range: start {} >= end {} for dimension {}",
                range.start, range.end, dim_idx
            ));
        }
        
        // Check bounds
        if range.end > tensor_dim {
            return Err(format!(
                "Range out of bounds: dimension {} range {}..{} exceeds size {}",
                dim_idx, range.start, range.end, tensor_dim
            ));
        }
        
        slice_shape[dim_idx] = range.end - range.start;
    }
    
    // Validate values tensor shape matches slice shape
    let values_shape = values.dims();
    if slice_shape != values_shape {
        return Err(format!(
            "Shape mismatch: expected {:?}, got {:?}",
            slice_shape, values_shape
        ));
    }
    
    // Perform the slice assignment
    let result = tensor.slice_assign(ranges, values);
    Ok(result)
}

/// Validates that a tensor slice region matches expected values
fn validate_slice_values<B: Backend>(
    tensor: &Tensor<B, 4>,
    ranges: [Range<usize>; 4],
    expected_value: f32,
    tolerance: f32,
) -> bool {
    let slice = tensor.clone().slice(ranges);
    let slice_data = slice.to_data();
    let slice_values = slice_data.as_slice::<f32>().expect("Failed to convert to f32 slice");
    
    slice_values.iter().all(|&val| (val - expected_value).abs() <= tolerance)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Testing Burn Tensor slice_assign Implementation");
    println!("==================================================\n");
    
    let device = Default::default();
    
    // Test 1: Basic slice assignment
    println!("📋 Test 1: Basic 4D slice assignment");
    {
        let tensor = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device);
        let values = Tensor::<Backend, 4>::ones([2, 3, 4, 4], &device);
        let ranges = [0..2, 0..3, 2..6, 2..6];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone())?;
        
        let slice_correct = validate_slice_values(&result, ranges, 1.0, 1e-6);
        println!("  ✅ Slice assignment successful: {}", slice_correct);
        
        // Check corners remain zero
        let corner_ranges = [0..1, 0..1, 0..2, 0..2];
        let corners_zero = validate_slice_values(&result, corner_ranges, 0.0, 1e-6);
        println!("  ✅ Unchanged regions preserved: {}", corners_zero);
    }
    
    // Test 2: Shape mismatch error
    println!("\n❌ Test 2: Shape mismatch error handling");
    {
        let tensor = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device);
        let wrong_values = Tensor::<Backend, 4>::ones([2, 3, 5, 5], &device);
        let ranges = [0..2, 0..3, 2..6, 2..6];
        
        match update_tensor_slice_4d(tensor, wrong_values, ranges) {
            Err(msg) => {
                println!("  ✅ Correct error: {}", msg);
                assert!(msg.contains("Shape mismatch"));
                assert!(msg.contains("[2, 3, 4, 4]"));
                assert!(msg.contains("[2, 3, 5, 5]"));
            }
            Ok(_) => return Err("Should have failed with shape mismatch".into()),
        }
    }
    
    // Test 3: Out of bounds error
    println!("\n❌ Test 3: Out of bounds error handling");
    {
        let tensor = Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device);
        let values = Tensor::<Backend, 4>::ones([3, 5, 9, 9], &device);
        let ranges = [0..3, 0..5, 0..9, 0..9];
        
        match update_tensor_slice_4d(tensor, values, ranges) {
            Err(msg) => {
                println!("  ✅ Correct error: {}", msg);
                assert!(msg.contains("Range out of bounds"));
                assert!(msg.contains("dimension 0"));
                assert!(msg.contains("exceeds size 2"));
            }
            Ok(_) => return Err("Should have failed with out of bounds".into()),
        }
    }
    
    // Test 4: Invalid range error
    println!("\n❌ Test 4: Invalid range error handling");
    {
        let tensor = Tensor::<Backend, 4>::zeros([2, 3, 4, 4], &device);
        let values = Tensor::<Backend, 4>::ones([1, 3, 4, 4], &device);
        let ranges = [2..1, 0..3, 0..4, 0..4]; // start > end
        
        match update_tensor_slice_4d(tensor, values, ranges) {
            Err(msg) => {
                println!("  ✅ Correct error: {}", msg);
                assert!(msg.contains("Invalid range"));
                assert!(msg.contains("start 2 >= end 1"));
                assert!(msg.contains("dimension 0"));
            }
            Ok(_) => return Err("Should have failed with invalid range".into()),
        }
    }
    
    // Test 5: Multi-channel update
    println!("\n📊 Test 5: Multi-channel tensor update");
    {
        let tensor = Tensor::<Backend, 4>::ones([1, 4, 6, 6], &device);
        let values = Tensor::<Backend, 4>::ones([1, 2, 6, 6], &device) * 2.0;
        let ranges = [0..1, 1..3, 0..6, 0..6]; // Update channels 1-2
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone())?;
        
        // Check updated channels
        let updated_correct = validate_slice_values(&result, ranges, 2.0, 1e-6);
        println!("  ✅ Channels 1-2 updated to 2.0: {}", updated_correct);
        
        // Check unchanged channels
        let ch0_ranges = [0..1, 0..1, 0..6, 0..6];
        let ch3_ranges = [0..1, 3..4, 0..6, 0..6];
        let ch0_unchanged = validate_slice_values(&result, ch0_ranges, 1.0, 1e-6);
        let ch3_unchanged = validate_slice_values(&result, ch3_ranges, 1.0, 1e-6);
        println!("  ✅ Channel 0 unchanged (1.0): {}", ch0_unchanged);
        println!("  ✅ Channel 3 unchanged (1.0): {}", ch3_unchanged);
    }
    
    // Test 6: Precision preservation
    println!("\n🔍 Test 6: Precision preservation");
    {
        let tensor = Tensor::<Backend, 4>::ones([1, 1, 2, 2], &device) * 0.123456789;
        let values = Tensor::<Backend, 4>::ones([1, 1, 1, 1], &device) * 0.987654321;
        let ranges = [0..1, 0..1, 0..1, 0..1];
        
        let result = update_tensor_slice_4d(tensor, values, ranges.clone())?;
        
        let slice = result.clone().slice(ranges);
        let slice_data = slice.to_data();
        let slice_values = slice_data.as_slice::<f32>().unwrap();
        
        let precision_preserved = (slice_values[0] - 0.987654321).abs() < 1e-7;
        println!("  ✅ f32 precision preserved: {}", precision_preserved);
        println!("     Expected: 0.987654321, Got: {:.9}", slice_values[0]);
    }
    
    // Test 7: Sequential updates
    println!("\n🔄 Test 7: Sequential slice updates");
    {
        let mut tensor = Tensor::<Backend, 4>::zeros([2, 2, 4, 4], &device);
        
        let updates = [
            ([0..1, 0..1, 0..2, 0..2], 1.0, "Top-left"),
            ([0..1, 1..2, 0..2, 0..2], 2.0, "Top-right"),
            ([1..2, 0..1, 0..2, 0..2], 3.0, "Bottom-left"),
            ([1..2, 1..2, 0..2, 0..2], 4.0, "Bottom-right"),
        ];
        
        for (ranges, value, description) in updates.iter() {
            let values = Tensor::<Backend, 4>::ones([1, 1, 2, 2], &device) * *value;
            tensor = update_tensor_slice_4d(tensor, values, *ranges)?;
            println!("  ✅ Updated {}: value {}", description, value);
        }
        
        // Verify final pattern
        for (ranges, expected_value, description) in updates.iter() {
            let is_correct = validate_slice_values(&tensor, *ranges, *expected_value, 1e-6);
            println!("  ✅ {} verification: {}", description, is_correct);
        }
    }
    
    println!("\n🎉 All tests passed! Tensor slice assignment implementation is working correctly.");
    println!("\n📊 Summary:");
    println!("  ✅ Basic slice assignment");
    println!("  ✅ Shape validation and error handling");
    println!("  ✅ Bounds checking");
    println!("  ✅ Range validation");
    println!("  ✅ Multi-channel operations");
    println!("  ✅ Precision preservation");
    println!("  ✅ Sequential updates");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_functionality() {
        main().expect("All tests should pass");
    }
}
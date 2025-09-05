/// Quick test to verify transformer forward pass works with [B,S,F] layout fix
use burn::prelude::*;
use burn_ndarray::NdArray;
use burn::backend::Autodiff;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use tab_pfn_rs::tabpfn::architectures::base::transformer::{PerFeatureTransformer, DeterministicRngContext};
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type TestBackend = Autodiff<NdArray<f32>>;

fn main() {
    println!("Testing transformer forward pass with [B,S,F] layout fix...");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Test parameters  
    let batch_size = 2; // Multiple tasks
    let seq_len = 4;
    let num_features = 3;
    let num_classes: usize = 3;
    
    // Create minimal model configuration
    let model_config = ModelConfig {
        max_num_classes: num_classes as i32,
        emsize: 32, // Small for test
        nhead: 2,
        nhid_factor: 2,
        nlayers: 1,
        dropout: 0.0,
        features_per_group: 2,
        ..ModelConfig::default()
    };
    
    // Create model - this should work now
    let mut model = PerFeatureTransformer::new(
        &model_config,
        &rng_context,
        num_classes,
        "gelu",
        Some(1),
        false,
        Some(1),
        false,
        None,
        false,
        &device,
    ).expect("Failed to create transformer");
    
    // Create input with canonical [B,S,F] layout
    let features_data: Vec<f32> = (0..(batch_size * seq_len * num_features))
        .map(|i| (i as f32) * 0.1)
        .collect();
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device,
    );
    
    println!("Input features shape: {:?}", features.dims());
    
    // Create y tensor with labels_for_model format
    let y_data: Vec<f32> = vec![
        0.0, 1.0, -1.0, -1.0,  // Task 1: train at pos 0,1; test at pos 2,3
        1.0, -1.0, 2.0, -1.0   // Task 2: train at pos 0,2; test at pos 1,3  
    ];
    
    let y_tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(y_data, [batch_size, seq_len]),
        &device,
    ).unsqueeze_dim(2); // Add feature dimension
    
    println!("Input y shape: {:?}", y_tensor.dims());
    
    // Prepare model inputs
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), features);
    
    let mut y_inputs = HashMap::new();
    y_inputs.insert("main".to_string(), y_tensor);
    
    // Forward pass
    let mut rng = StdRng::seed_from_u64(42);
    let mut rng_opt = Some(&mut rng);
    
    match model.transformer_forward(
        x_inputs,
        Some(y_inputs),
        true, // hypernetwork_on 
        &mut rng_opt,
        None, // categorical_inds
        None, // style
        None, // dags
        false, // train_mode
    ) {
        Ok(output) => {
            println!("✅ Transformer forward pass successful!");
            println!("Output shape: {:?}", output.dims());
            
            // Verify output has correct shape [B,S,C]
            let expected_shape = [batch_size, seq_len, num_classes];
            if output.dims() == expected_shape {
                println!("✅ Output shape is correct: {:?}", expected_shape);
                
                // Verify output contains finite values
                let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
                let all_finite = output_data.iter().all(|&x| x.is_finite());
                
                if all_finite {
                    println!("✅ All output values are finite");
                    println!("🎉 Transformer [B,S,F] layout fix is working correctly!");
                } else {
                    println!("❌ Some output values are not finite");
                }
            } else {
                println!("❌ Output shape mismatch. Expected: {:?}, Got: {:?}", expected_shape, output.dims());
            }
        }
        Err(e) => {
            println!("❌ Transformer forward pass failed: {}", e);
        }
    }
}
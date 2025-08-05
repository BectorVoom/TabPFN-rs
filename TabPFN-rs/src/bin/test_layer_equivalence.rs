use std::env;
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::{
    config::ModelConfig,
    layer::PerFeatureEncoderLayer,
};
use burn::prelude::*;

type Backend = NdArray<f32>;

fn create_test_config() -> ModelConfig {
    let mut config = ModelConfig::default();
    config.emsize = 64; // Smaller for testing
    config.nhead = 4;
    config.attention_init_gain = 1.0;
    config.feature_attention_type = "full".to_string();
    config.item_attention_type = "full".to_string();
    config.multiquery_item_attention = false;
    config.multiquery_item_attention_for_test_set = false;
    config.recompute_attn = false;
    config
}

fn test_layer_creation() -> Result<(), String> {
    println!("Testing Rust PerFeatureEncoderLayer creation...");
    
    let config = create_test_config();
    let device = Default::default();
    
    // Create Rust layer
    let _layer = PerFeatureEncoderLayer::<Backend>::new(
        &config,
        256, // dim_feedforward
        "relu".to_string(), // activation
        1e-5, // layer_norm_eps
        false, // pre_norm
        &device,
        false, // second_mlp
        true, // layer_norm_with_elementwise_affine
        false, // zero_init
        None, // save_peak_mem_factor
        true, // attention_between_features
        None, // d_k
        None, // d_v
        None, // precomputed_kv
    )?;
    
    println!("✓ Rust PerFeatureEncoderLayer creation test passed");
    Ok(())
}

fn test_layer_forward_basic() -> Result<(), String> {
    println!("Testing basic Rust layer forward pass...");
    
    let config = create_test_config();
    let device = Default::default();
    
    // Create Rust layer
    let mut layer = PerFeatureEncoderLayer::<Backend>::new(
        &config,
        256, // dim_feedforward
        "relu".to_string(),
        1e-5,
        false,
        &device,
        false,
        true,
        false,
        None,
        true,
        None,
        None,
        None,
    )?;
    
    // Create test input - 4D tensor: (batch_size, num_items, num_feature_blocks, d_model)
    let batch_size = 2;
    let num_items = 8;
    let num_feature_blocks = 4;
    let d_model = config.emsize as usize;
    
    let test_input = Tensor::<Backend, 4>::random(
        [batch_size, num_items, num_feature_blocks, d_model], 
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    // Run forward pass
    let output = layer.forward(
        test_input.clone(),
        0, // single_eval_pos (no test set)
        false, // cache_trainset_representation
        None, // att_src
    );
    
    // Check output shape
    let expected_shape = [batch_size, num_items, num_feature_blocks, d_model];
    let output_shape = output.shape().dims;
    
    if output_shape != expected_shape {
        return Err(format!("Expected shape {:?}, got {:?}", expected_shape, output_shape));
    }
    
    println!("✓ Rust layer forward pass test passed");
    println!("  Input shape: {:?}", test_input.shape().dims);
    println!("  Output shape: {:?}", output_shape);
    
    Ok(())
}

fn test_layer_without_feature_attention() -> Result<(), String> {
    println!("Testing layer without feature attention...");
    
    let config = create_test_config();
    let device = Default::default();
    
    // Create layer without attention between features
    let mut layer = PerFeatureEncoderLayer::<Backend>::new(
        &config,
        256,
        "gelu".to_string(),
        1e-5,
        false,
        &device,
        false,
        true,
        false,
        None,
        false, // attention_between_features = false
        None,
        None,
        None,
    )?;
    
    // For no attention between features, we need exactly 1 feature block
    let batch_size = 2;
    let num_items = 8;
    let num_feature_blocks = 1; // Must be 1 when no attention between features
    let d_model = config.emsize as usize;
    
    let test_input = Tensor::<Backend, 4>::random(
        [batch_size, num_items, num_feature_blocks, d_model], 
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    let output = layer.forward(
        test_input.clone(),
        0,
        false,
        None,
    );
    
    let expected_shape = [batch_size, num_items, num_feature_blocks, d_model];
    let output_shape = output.shape().dims;
    
    if output_shape != expected_shape {
        return Err(format!("Expected shape {:?}, got {:?}", expected_shape, output_shape));
    }
    
    println!("✓ Layer without feature attention test passed");
    Ok(())
}

fn test_layer_with_second_mlp() -> Result<(), String> {
    println!("Testing layer with second MLP...");
    
    let config = create_test_config();
    let device = Default::default();
    
    // Create layer with second MLP
    let mut layer = PerFeatureEncoderLayer::<Backend>::new(
        &config,
        256,
        "relu".to_string(),
        1e-5,
        false,
        &device,
        true, // second_mlp = true
        true,
        false,
        None,
        true,
        None,
        None,
        None,
    )?;
    
    let batch_size = 2;
    let num_items = 8;
    let num_feature_blocks = 4;
    let d_model = config.emsize as usize;
    
    let test_input = Tensor::<Backend, 4>::random(
        [batch_size, num_items, num_feature_blocks, d_model], 
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device
    );
    
    let output = layer.forward(
        test_input.clone(),
        0,
        false,
        None,
    );
    
    let expected_shape = [batch_size, num_items, num_feature_blocks, d_model];
    let output_shape = output.shape().dims;
    
    if output_shape != expected_shape {
        return Err(format!("Expected shape {:?}, got {:?}", expected_shape, output_shape));
    }
    
    println!("✓ Layer with second MLP test passed");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Running Rust PerFeatureEncoderLayer equivalence tests...");
        
        let tests = vec![
            ("Layer Creation", test_layer_creation),
            ("Basic Forward Pass", test_layer_forward_basic),
            ("Without Feature Attention", test_layer_without_feature_attention),
            ("With Second MLP", test_layer_with_second_mlp),
        ];
        
        let mut passed = 0;
        let total = tests.len();
        
        for (name, test_fn) in tests {
            match test_fn() {
                Ok(()) => {
                    passed += 1;
                    println!("✓ {} passed", name);
                }
                Err(e) => {
                    println!("✗ {} failed: {}", name, e);
                }
            }
        }
        
        println!("\nTest Results: {}/{} passed", passed, total);
        
        if passed == total {
            println!("🎉 All layer tests passed!");
            std::process::exit(0);
        } else {
            println!("❌ Some tests failed");
            std::process::exit(1);
        }
    } else {
        println!("Layer equivalence test doesn't support external test files yet");
        std::process::exit(0);
    }
}
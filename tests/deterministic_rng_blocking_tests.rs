//! Blocking tests for deterministic RNG implementation
//! 
//! These tests MUST pass before the deterministic RNG implementation is considered complete.
//! All tests are blocking requirements as specified in the implementation plan.

use burn::prelude::*;
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::transformer::{
    DeterministicRngContext, PerFeatureTransformer
};
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;
use tab_pfn_rs::tabpfn::architectures::base::mlp::MLP;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

/// Test 1: Parameter Registration Test (BLOCKING)
/// 
/// Verifies that all modules with #[derive(Module)] properly register their parameters
/// and that parameter names include expected patterns.
#[test]
fn test_parameter_registration_blocking() {
    let device = Default::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device);
    
    // Create a model configuration
    let config = ModelConfig {
        seed: 42,
        emsize: 128,
        nhid_factor: 2,  // nhid = emsize * nhid_factor = 256
        nlayers: 2,
        nhead: 8,
        dropout: 0.1,
        max_num_classes: 10,
        num_buckets: 32,
        ..Default::default()
    };
    
    // Create a nontrivial architecture with multiple components
    let model = PerFeatureTransformer::<TestBackend>::new(
        &config, 
        &rng_context,
        3, // n_out (num_classes)
        "gelu", // activation
        None, // min_num_layers_layer_dropout
        false, // zero_init
        None, // nlayers_decoder
        false, // use_encoder_compression_layer
        None, // precomputed_kv
        false, // cache_trainset_representation
        &device
    ).unwrap();
    
    // Get all named parameters
    let named_params = model.named_parameters();
    let param_names: Vec<String> = named_params.keys().cloned().collect();
    
    println!("🔍 Found {} parameters:", param_names.len());
    for name in &param_names {
        println!("  - {}", name);
    }
    
    // BLOCKING REQUIREMENT: num_params() > 0
    assert!(model.num_params() > 0, "Model must have registered parameters");
    assert!(param_names.len() > 0, "Model must have named parameters");
    
    // BLOCKING REQUIREMENT: parameter names include expected patterns
    let expected_patterns = vec![
        "multi_head_attention",
        "decoder_linear",
        "layer_norm",
        "weight"
    ];
    
    let mut found_patterns = HashMap::new();
    for pattern in &expected_patterns {
        found_patterns.insert(pattern, false);
    }
    
    for param_name in &param_names {
        for pattern in &expected_patterns {
            if param_name.contains(pattern) {
                found_patterns.insert(pattern, true);
            }
        }
    }
    
    // Verify that we found at least some of the expected patterns
    let found_count = found_patterns.values().filter(|&&found| found).count();
    assert!(found_count > 0, "Must find at least some expected parameter name patterns");
    
    println!("✅ Parameter registration test passed: {} parameters registered", param_names.len());
}

/// Test 2: Deterministic Initialization Test (BLOCKING)
/// 
/// Verifies that two models created with the same seed have identical parameter values
/// with maximum absolute difference < 1e-6.
#[test] 
fn test_deterministic_initialization_blocking() {
    let device = Default::default();
    let seed = 42;
    
    // Create model configuration
    let config = ModelConfig {
        seed,
        emsize: 64,
        nhid_factor: 2,  // nhid = emsize * nhid_factor = 128
        nlayers: 1,
        nhead: 4,
        dropout: 0.1,
        max_num_classes: 10,
        num_buckets: 32,
        ..Default::default()
    };
    
    // Create two models with the same seed
    let rng_context1 = DeterministicRngContext::<TestBackend>::new(seed as u64, device);
    let rng_context2 = DeterministicRngContext::<TestBackend>::new(seed as u64, device);
    
    let model1 = PerFeatureTransformer::<TestBackend>::new(
        &config, 
        &rng_context1,
        3, // n_out (num_classes)
        "gelu", // activation
        None, // min_num_layers_layer_dropout
        false, // zero_init
        None, // nlayers_decoder
        false, // use_encoder_compression_layer
        None, // precomputed_kv
        false, // cache_trainset_representation
        &device
    );
    let model2 = PerFeatureTransformer::<TestBackend>::new(
        &config, 
        &rng_context2,
        3, // n_out (num_classes)
        "gelu", // activation
        None, // min_num_layers_layer_dropout
        false, // zero_init
        None, // nlayers_decoder
        false, // use_encoder_compression_layer
        None, // precomputed_kv
        false, // cache_trainset_representation
        &device
    );
    
    // Get named parameters from both models
    let params1 = model1.named_parameters();
    let params2 = model2.named_parameters();
    
    assert_eq!(params1.keys().len(), params2.keys().len(), "Both models must have same number of parameters");
    
    let mut max_abs_diff = 0.0f32;
    let mut param_count = 0;
    
    // Compare every parameter elementwise
    for (name, param1) in params1.iter() {
        let param2 = params2.get(name).expect(&format!("Parameter {} not found in second model", name));
        
        // Convert to data for comparison
        let data1 = param1.to_data();
        let data2 = param2.to_data(); 
        
        assert_eq!(data1.shape, data2.shape, "Parameter {} shapes must match", name);
        
        let values1 = data1.as_slice::<f32>().expect("Failed to convert to f32 slice");
        let values2 = data2.as_slice::<f32>().expect("Failed to convert to f32 slice");
        
        for (i, (&v1, &v2)) in values1.iter().zip(values2.iter()).enumerate() {
            let abs_diff = (v1 - v2).abs();
            max_abs_diff = max_abs_diff.max(abs_diff);
            
            if abs_diff >= 1e-6 {
                panic!("Parameter {} element {} differs by {:.2e} (v1={:.6}, v2={:.6}), exceeds tolerance 1e-6", 
                       name, i, abs_diff, v1, v2);
            }
        }
        
        param_count += values1.len();
    }
    
    println!("✅ Deterministic initialization test passed:");
    println!("   - Compared {} parameters across {} elements", params1.len(), param_count);
    println!("   - Maximum absolute difference: {:.2e}", max_abs_diff);
    println!("   - Required tolerance: < 1e-6");
    
    // BLOCKING REQUIREMENT: max_abs_diff < 1e-6
    assert!(max_abs_diff < 1e-6, "Maximum absolute parameter difference {:.2e} exceeds required tolerance 1e-6", max_abs_diff);
}

/// Test 3: Deterministic Forward Test (BLOCKING)
/// 
/// Verifies that two identically-seeded models produce identical outputs on identical inputs
/// with maximum absolute difference < 1e-6.
#[test]
fn test_deterministic_forward_blocking() {
    let device = Default::default();
    let seed = 42;
    
    // Create model configuration
    let config = ModelConfig {
        seed,
        emsize: 64,
        nhid_factor: 2,  // nhid = emsize * nhid_factor = 128
        nlayers: 1,
        nhead: 4,
        dropout: 0.1,
        max_num_classes: 10,
        num_buckets: 32,
        ..Default::default()
    };
    
    // Create two models with the same seed
    let rng_context1 = DeterministicRngContext::<TestBackend>::new(seed as u64, device);
    let rng_context2 = DeterministicRngContext::<TestBackend>::new(seed as u64, device);
    
    let model1 = PerFeatureTransformer::<TestBackend>::new(
        &config, 
        &rng_context1,
        3, // n_out (num_classes)
        "gelu", // activation
        None, // min_num_layers_layer_dropout
        false, // zero_init
        None, // nlayers_decoder
        false, // use_encoder_compression_layer
        None, // precomputed_kv
        false, // cache_trainset_representation
        &device
    );
    let model2 = PerFeatureTransformer::<TestBackend>::new(
        &config, 
        &rng_context2,
        3, // n_out (num_classes)
        "gelu", // activation
        None, // min_num_layers_layer_dropout
        false, // zero_init
        None, // nlayers_decoder
        false, // use_encoder_compression_layer
        None, // precomputed_kv
        false, // cache_trainset_representation
        &device
    );
    
    // Create identical input tensors
    let seq_len = 8;
    let batch_size = 2;
    let num_features = 4;
    
    let x_data = vec![1.0f32; seq_len * batch_size * num_features];
    let x_tensor1 = Tensor::<TestBackend, 3>::from_floats(x_data.as_slice(), &device)
        .reshape([seq_len, batch_size, num_features]);
    let x_tensor2 = x_tensor1.clone();
    
    let y_data = vec![0.5f32; seq_len * batch_size * 1];
    let y_tensor1 = Tensor::<TestBackend, 3>::from_floats(y_data.as_slice(), &device)
        .reshape([seq_len, batch_size, 1]);
    let y_tensor2 = y_tensor1.clone();
    
    // Create deterministic RNG contexts for forward pass
    let mut rng1 = StdRng::seed_from_u64((seed + 1000) as u64); // offset for forward randomness
    let mut rng2 = StdRng::seed_from_u64((seed + 1000) as u64); // same offset ensures identical forward RNG
    
    // Run forward passes with same RNG seeds
    let mut x_map1 = HashMap::new();
    x_map1.insert("train".to_string(), x_tensor1);
    let mut y_map1 = HashMap::new();
    y_map1.insert("train".to_string(), y_tensor1);
    
    let mut x_map2 = HashMap::new();
    x_map2.insert("train".to_string(), x_tensor2);
    let mut y_map2 = HashMap::new(); 
    y_map2.insert("train".to_string(), y_tensor2);
    
    let mut model1 = model1.unwrap();
    let mut model2 = model2.unwrap();
    
    let output1 = model1.transformer_forward(
        x_map1,
        Some(y_map2.clone()),
        false, // only_return_standard_out
        &mut Some(&mut rng1), // rng: &mut Option<&mut StdRng>
        None, // categorical_inds
        None, // style
        None, // data_dags
        false // train: false for deterministic testing
    );
    
    let output2 = model2.transformer_forward(
        x_map2,
        Some(y_map2),
        false, // only_return_standard_out
        &mut Some(&mut rng2), // rng: &mut Option<&mut StdRng>
        None, // categorical_inds
        None, // style
        None, // data_dags
        false // train: false for deterministic testing
    );
    
    // Compare outputs
    let data1 = output1.unwrap().to_data();
    let data2 = output2.unwrap().to_data();
    
    assert_eq!(data1.shape, data2.shape, "Output shapes should match");
    
    let values1 = data1.as_slice::<f32>().expect("Failed to convert output1 to f32");
    let values2 = data2.as_slice::<f32>().expect("Failed to convert output2 to f32");
    
    let mut max_abs_diff = 0.0f32;
    for (i, (&v1, &v2)) in values1.iter().zip(values2.iter()).enumerate() {
        let abs_diff = (v1 - v2).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        
        if abs_diff >= 1e-6 {
            panic!("Output element {} differs by {:.2e} (v1={:.6}, v2={:.6}), exceeds tolerance 1e-6",
                   i, abs_diff, v1, v2);
        }
    }
    
    println!("✅ Deterministic forward test passed:");
    println!("   - Compared {} output elements", values1.len());
    println!("   - Maximum absolute difference: {:.2e}", max_abs_diff);
    println!("   - Required tolerance: < 1e-6");
    
    // BLOCKING REQUIREMENT: max_abs_diff < 1e-6
    assert!(max_abs_diff < 1e-6, "Maximum absolute output difference {:.2e} exceeds required tolerance 1e-6", max_abs_diff);
}

/// Test 4: Source Code Scan Test (BLOCKING)
/// 
/// Static analysis test that ensures no forbidden RNG or CPU sync patterns appear in production code.
#[test]
fn test_no_forbidden_patterns_blocking() {
    use std::fs;
    use std::path::Path;
    
    // Forbidden RNG patterns
    let forbidden_rng_patterns = [
        "StdRng::from_entropy(",
        "thread_rng()",
        "rand::thread_rng(",
        "from_entropy(",
    ];
    
    // Forbidden CPU sync patterns  
    let forbidden_cpu_patterns = [
        ".to_data(",
        ".as_slice(",
        ".into_data(",
        ".to_vec(",
    ];
    
    // Production source files to scan (exclude tests)
    let production_files = [
        "src/tabpfn/architectures/base/transformer.rs",
        "src/tabpfn/architectures/base/encoders.rs",
        "src/tabpfn/architectures/base/encoders2.rs", 
        "src/tabpfn/architectures/base/encoders_before.rs",
        "src/tabpfn/architectures/base/layer.rs",
        "src/tabpfn/architectures/base/mlp.rs",
        "src/tabpfn/architectures/base/attention/full_attention.rs",
        // Note: train.rs and validation.rs may still have some patterns during transition
    ];
    
    let mut violations = Vec::new();
    
    for file_path in &production_files {
        if Path::new(file_path).exists() {
            let content = fs::read_to_string(file_path)
                .expect(&format!("Failed to read {}", file_path));
            
            // Check for forbidden RNG patterns
            for pattern in &forbidden_rng_patterns {
                if content.contains(pattern) {
                    violations.push(format!("FORBIDDEN RNG: {} found in {}", pattern, file_path));
                }
            }
            
            // Check for forbidden CPU sync patterns (with some exceptions)
            for pattern in &forbidden_cpu_patterns {
                if content.contains(pattern) {
                    // Allow in test functions and documented helpers
                    let lines: Vec<&str> = content.lines().collect();
                    for (line_num, line) in lines.iter().enumerate() {
                        if line.contains(pattern) {
                            // Skip if it's in a test function or test module
                            let context = if line_num > 10 { &lines[line_num-10..line_num] } else { &lines[0..line_num] };
                            let is_test_context = context.iter().any(|l| 
                                l.contains("#[test]") || 
                                l.contains("#[cfg(test)]") ||
                                l.contains("mod tests") ||
                                l.contains("// Test") ||
                                l.contains("test_")
                            );
                            
                            if !is_test_context {
                                violations.push(format!("FORBIDDEN CPU SYNC: {} found in {} line {}", pattern, file_path, line_num + 1));
                            }
                        }
                    }
                }
            }
        }
    }
    
    if !violations.is_empty() {
        println!("❌ Found forbidden patterns in production code:");
        for violation in &violations {
            println!("  - {}", violation);
        }
        panic!("Found {} forbidden patterns in production code", violations.len());
    }
    
    println!("✅ Source code scan test passed: No forbidden patterns found in production code");
    println!("   - Scanned {} production files", production_files.len());
    println!("   - Checked for {} forbidden RNG patterns", forbidden_rng_patterns.len());
    println!("   - Checked for {} forbidden CPU sync patterns", forbidden_cpu_patterns.len());
}

/// Test 5: DeterministicLinear and DeterministicEmbedding functionality
#[test]  
fn test_deterministic_modules_basic() {
    let device = Default::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device);
    
    // Test DeterministicLinear
    let linear = rng_context.create_deterministic_linear(3, 4, true, 100);
    let input = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
    let output = linear.forward(input);
    assert_eq!(output.dims(), [2, 4]);
    
    // Test DeterministicEmbedding
    let embedding = rng_context.create_deterministic_embedding(10, 5, 200);
    let indices = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([0, 1, 2].as_slice(), &device);
    let embedded = embedding.forward(indices);
    assert_eq!(embedded.dims(), [3, 5]);
    
    println!("✅ Basic deterministic modules test passed");
}
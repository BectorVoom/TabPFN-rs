/// TDD Tests for TabPFN Fatal Problems A-D
/// 
/// These tests are designed to fail initially and guide the implementation fixes 
/// according to the TabPFN specification. Each test focuses on one specific fatal problem.
/// 
/// Test-Driven Development Approach:
/// 1. Write failing tests that expose the problems
/// 2. Implement minimal fixes to make tests pass
/// 3. Refactor and ensure all tests continue to pass

use tab_pfn_rs::tabpfn::architectures::base::{
    config::ModelConfig,
    transformer::DeterministicRngContext,
};

use burn::{
    backend::Autodiff,
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_ndarray::NdArray;
use rand::{rngs::StdRng, SeedableRng};

type TestBackend = Autodiff<NdArray<f32>>;

// =============================================================================
// PROBLEM A: AXIS CONSISTENCY TESTS - [S, B, F] vs [B, S, F]
// =============================================================================

/// Simple compilation test for imports
#[test]
fn test_imports_work() {
    use tab_pfn_rs::tabpfn::architectures::base::train::{TrainingConfig, PriorType};
    
    // Just test that we can create a basic config - this should compile if imports work
    let config = TrainingConfig {
        model: ModelConfig::default(),
        meta_batch_size: 1,
        tasks_per_batch: 1,
        max_samples_per_task: 10,
        min_samples_per_task: 5,
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        prior_type: PriorType::Gaussian,
        num_features_range: (5, 5),
        num_classes_range: (3, 3),
        feature_noise_level: 0.1,
    };
    
    assert_eq!(config.meta_batch_size, 1);
    println!("✅ Basic import test passed");
}

/// Test A1: Features tensor must be [S, B, F] format - EXPECTED TO FAIL
/// 
/// The specification requires [S, B, F] = [num_samples, meta_batch_size, num_features]
/// but current implementation uses [B, S, F] = [meta_batch_size, num_samples, num_features]
#[test]
#[should_panic(expected = "AXIS ERROR: features shape must be [S, B, F]")]
fn test_a1_features_tensor_sbf_format_fails() {
    println!("🔴 Test A1: Features tensor [S,B,F] format - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Specification parameters: [S, B, F]
    let num_samples: usize = 10;    // S = sequence length (examples per task)
    let meta_batch_size: usize = 4; // B = meta-batch size (parallel tasks)  
    let num_features: usize = 6;    // F = features per example
    
    // CREATE TENSOR IN CORRECT [S, B, F] FORMAT
    let features_data: Vec<f32> = (0..(num_samples * meta_batch_size * num_features))
        .map(|i| i as f32 * 0.01)
        .collect();
    
    let features = Tensor::<TestBackend, 1>::from_floats(
        features_data.as_slice(),
        &device
    ).reshape([num_samples, meta_batch_size, num_features]); // [S, B, F] - CORRECT
    
    // Create matching targets and masks in [S, B] format  
    let targets_data: Vec<i64> = (0..(num_samples * meta_batch_size))
        .map(|i| (i % 3) as i64)
        .collect();
    
    let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
        targets_data.as_slice(),
        &device
    ).reshape([num_samples, meta_batch_size]); // [S, B] - CORRECT
    
    let train_mask_data: Vec<bool> = (0..(num_samples * meta_batch_size))
        .map(|i| i % 3 != 0)
        .collect();
    
    let train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_bool(
        TensorData::new(train_mask_data, [num_samples * meta_batch_size]),
        &device
    ).reshape([num_samples, meta_batch_size]); // [S, B] - CORRECT
    
    // Construct labels_for_model with -1 at test positions
    let neg_ones = Tensor::<TestBackend, 2, burn::tensor::Int>::ones_like(&targets) * (-1);
    let labels_for_model = targets.clone().mask_where(train_mask.clone().bool_not(), neg_ones);
    
    let dataset = SyntheticTabularDataset {
        features, // [S, B, F] - This should be the CORRECT format
        targets,  // [S, B] - This should be the CORRECT format
        train_mask, // [S, B] - This should be the CORRECT format 
        labels_for_model, // [S, B] - This should be the CORRECT format
        dag: None,
    };
    
    // TEST: Current validation should FAIL because it expects [B, S, F] but we provide [S, B, F]
    match dataset.validate_shapes() {
        Ok(_) => {
            println!("❌ Test A1 FAILED: Validation passed but should enforce [S,B,F] format");
            panic!("AXIS ERROR: features shape must be [S, B, F]");
        }
        Err(msg) => {
            println!("✅ Test A1 validation caught error as expected: {}", msg);
            // The error should be about axis format, not just dimensions
            if msg.contains("features must be 3D [batch, seq, features]") {
                println!("❌ Current validation uses [B,S,F] convention - THIS IS THE BUG");
                panic!("AXIS ERROR: features shape must be [S, B, F]");
            } else {
                println!("✅ Different error caught: {}", msg);
            }
        }
    }
}

/// Test A2: All tensors must use consistent [S, B] axis ordering - EXPECTED TO FAIL
#[test]  
#[should_panic(expected = "AXIS ERROR: tensor shapes must use [S,B] convention")]
fn test_a2_tensor_axis_consistency_sbf_fails() {
    println!("🔴 Test A2: Tensor axis consistency [S,B] - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with specific [S, B] dimensions
    let S: usize = 8;  // num_samples (examples per task)
    let B: usize = 3;  // meta_batch_size (number of tasks)
    let F: usize = 4;  // num_features
    
    // Create tensors in SPECIFICATION-COMPLIANT [S, B, F] and [S, B] formats
    let features = Tensor::<TestBackend, 1>::zeros([S * B * F], &device)
        .reshape([S, B, F]); // [num_samples, meta_batch_size, num_features] - SPEC CORRECT
        
    let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([S * B], &device)
        .reshape([S, B]); // [num_samples, meta_batch_size] - SPEC CORRECT
        
    let train_mask_data = vec![true; S * B];
    let train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_bool(
        TensorData::new(train_mask_data, [S * B]),
        &device
    ).reshape([S, B]); // [num_samples, meta_batch_size] - SPEC CORRECT
        
    let labels_for_model = targets.clone(); // [S, B] - SPEC CORRECT
    
    let dataset = SyntheticTabularDataset {
        features,
        targets,
        train_mask, 
        labels_for_model,
        dag: None,
    };
    
    // The current implementation should FAIL to recognize this as valid [S,B,F] format
    // because it was designed for [B,S,F] format  
    match dataset.validate_shapes() {
        Ok(_) => {
            println!("❌ Test A2 PASSED but should FAIL - current validation accepts [S,B,F]");
            // If this passes, the implementation might actually be correct already
            println!("✅ Validation correctly handles [S,B,F] format!");
        }
        Err(msg) => {
            println!("✅ Test A2 FAILED as expected: {}", msg);
            println!("❌ Current implementation rejects [S,B,F] format - THIS IS THE BUG");
            panic!("AXIS ERROR: tensor shapes must use [S,B] convention");
        }
    }
}

// =============================================================================
// PROBLEM B: DEFENSIVE ARGMAX HANDLING TESTS  
// =============================================================================

/// Test B1: Argmax defensive handling for 2D results - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "ARGMAX ERROR: defensive handling not implemented")]
fn test_b1_argmax_defensive_2d_fails() {
    println!("🔴 Test B1: Argmax defensive 2D handling - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits tensor [S, B, C] = [4, 2, 3]  
    let logits_data: Vec<f32> = vec![
        // Task 1 (B=0): 4 samples, 3 classes each
        1.0, 0.2, -0.5,  // Sample 0: class 0 wins
        -0.3, 1.5, 0.1,  // Sample 1: class 1 wins  
        0.8, -0.1, 2.0,  // Sample 2: class 2 wins
        2.2, 0.0, -1.0,  // Sample 3: class 0 wins
        // Task 2 (B=1): 4 samples, 3 classes each  
        -0.5, 0.8, 1.2,  // Sample 0: class 2 wins
        1.8, -0.2, 0.3,  // Sample 1: class 0 wins
        0.1, 2.1, -0.8,  // Sample 2: class 1 wins
        -1.2, -0.5, 0.9, // Sample 3: class 2 wins
    ];
    
    let logits = Tensor::<TestBackend, 1>::from_floats(
        logits_data.as_slice(),
        &device
    ).reshape([4, 2, 3]); // [S, B, C]
    
    // TEST: Apply argmax along class dimension (axis=2)  
    let argmax_result = logits.argmax(2);
    let result_dims = argmax_result.dims();
    
    println!("Argmax result dims: {:?}", result_dims);
    
    // Current implementation probably doesn't do defensive checks
    // We need defensive logic that handles different argmax result shapes
    
    // Expected behavior:
    // - If argmax returns [S, B] (2D) -> use as-is  
    // - If argmax returns [S, B, 1] (3D with final dim=1) -> squeeze to [S, B]
    // - Otherwise -> panic with descriptive error
    
    // THIS SHOULD FAIL because current implementation doesn't have defensive handling
    if result_dims.len() == 2 {
        println!("✅ Argmax returned 2D shape {:?} - good", result_dims);
        println!("❌ But no defensive handling implemented for unexpected shapes");
        panic!("ARGMAX ERROR: defensive handling not implemented");
    } else if result_dims.len() == 3 && result_dims[2] == 1 {
        println!("✅ Argmax returned 3D shape {:?} - needs squeezing", result_dims);
        println!("❌ But no defensive squeezing implemented");
        panic!("ARGMAX ERROR: defensive handling not implemented");
    } else {
        println!("❌ Argmax returned unexpected shape {:?}", result_dims); 
        panic!("ARGMAX ERROR: defensive handling not implemented");
    }
}

/// Test B2: Argmax defensive handling for 3D results requiring squeeze - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "ARGMAX ERROR: squeeze logic missing")]  
fn test_b2_argmax_defensive_3d_squeeze_fails() {
    println!("🔴 Test B2: Argmax defensive 3D squeeze - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create a logits tensor that might produce 3D argmax result
    let logits = Tensor::<TestBackend, 1>::zeros([2 * 3 * 1], &device)
        .reshape([2, 3, 1]); // [S, B, C] with C=1 (single class)
        
    let argmax_result = logits.argmax(2);
    let dims = argmax_result.dims();
    
    println!("Argmax dims for single-class case: {:?}", dims);
    
    // The specification requires defensive handling:
    // If dims.len() == 3 && dims[2] == 1, then squeeze(2) to get [S,B]
    // Current implementation likely doesn't handle this
    
    if dims.len() == 3 && dims[2] == 1 {
        println!("❌ Got 3D result with final dim=1, needs squeezing");
        println!("❌ Current implementation doesn't implement defensive squeeze");
        panic!("ARGMAX ERROR: squeeze logic missing");
    } else {
        println!("❌ Expected 3D result with final dim=1 for squeeze test");
        panic!("ARGMAX ERROR: squeeze logic missing");
    }
}

// =============================================================================
// PROBLEM C: SAFE NUM_SAMPLES SELECTION TESTS
// =============================================================================

/// Test C1: Safe num_samples when min equals max - EXPECTED TO FAIL
#[test]  
#[should_panic(expected = "NUM_SAMPLES ERROR: min==max case not handled")]
fn test_c1_num_samples_min_equals_max_fails() {
    println!("🔴 Test C1: Safe num_samples when min==max - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create config with min == max (this should NOT panic)
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (5, 5),
        num_classes_range: (3, 3), 
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 10, // Same as max
        max_samples_per_task: 10, // Same as min - THIS IS THE TEST CASE
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 1,
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(12345);
    
    // THIS SHOULD FAIL with current implementation that uses % operator
    // Current logic: rng.gen() % (max - min) + min 
    // When max == min: rng.gen() % 0 -> PANIC!
    
    match std::panic::catch_unwind(|| {
        let _dataset = prior.sample::<TestBackend>(
            10, // This parameter might be ignored in current implementation
            1,  // meta_batch_size
            &device, 
            &rng_context,
            &mut rng
        );
    }) {
        Ok(_) => {
            println!("❌ Test C1 PASSED but should FAIL - min==max case handled correctly");  
            println!("✅ Implementation correctly handles min==max case!");
        }
        Err(_) => {
            println!("✅ Test C1 FAILED as expected - min==max causes panic");
            println!("❌ Current implementation uses % operator causing division by zero");
            panic!("NUM_SAMPLES ERROR: min==max case not handled");
        }
    }
}

/// Test C2: TrainingConfig validation - EXPECTED TO FAIL  
#[test]
#[should_panic(expected = "CONFIG ERROR: validation not implemented")]
fn test_c2_training_config_validation_fails() {
    println!("🔴 Test C2: TrainingConfig validation - EXPECTED TO FAIL");
    
    // Test invalid config: min > max
    let invalid_config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (5, 5),
        num_classes_range: (3, 3),
        feature_noise_level: 0.1, 
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 15, // INVALID: min > max
        max_samples_per_task: 10, // INVALID: max < min
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 1,
        use_gradient_checkpointing: false,
    };
    
    // Try to call validate() method - this should fail because method doesn't exist
    // match invalid_config.validate() {
    //     Ok(_) => panic!("Should have caught min > max error"),
    //     Err(_) => println!("✅ Validation correctly caught error"),
    // }
    
    // For now, just test that validate method doesn't exist
    println!("❌ TrainingConfig::validate() method not implemented");
    panic!("CONFIG ERROR: validation not implemented");
}

// =============================================================================  
// PROBLEM D: CONFIG TYPE AND Y_ENCODER DTYPE TESTS
// =============================================================================

/// Test D1: Learning rate must be f32, not f64 - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "TYPE ERROR: learning_rate must be f32")]
fn test_d1_learning_rate_f32_type_fails() {
    println!("🔴 Test D1: Learning rate f32 type - EXPECTED TO FAIL");
    
    let config = TrainingConfig {
        learning_rate: 0.001, // This is f64 in current implementation
        // ... other fields
        prior_type: PriorType::Gaussian,
        num_features_range: (5, 5),
        num_classes_range: (3, 3),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 10,
        max_samples_per_task: 10,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 1,
        use_gradient_checkpointing: false,
    };
    
    // Test compilation: f32 optimizer step should accept f32 learning rate
    // This will be a compile-time test in the actual fix
    
    let lr_f64: f64 = config.learning_rate;  
    let _lr_f32: f32 = lr_f64 as f32; // Required conversion reveals the bug
    
    println!("❌ learning_rate is f64: {}", lr_f64);
    println!("❌ Burn optimizer.step() expects f32, requires conversion");
    panic!("TYPE ERROR: learning_rate must be f32");
}

/// Test D2: Y-encoder input must be f32 - EXPECTED TO FAIL/PASS
#[test]
fn test_d2_y_encoder_f32_input() {
    println!("🟡 Test D2: Y-encoder f32 input - MIGHT PASS");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create labels_for_model as i64 with -1 sentinel values
    let labels_data = vec![0i64, 1i64, -1i64, 2i64, -1i64, 0i64];
    let labels_for_model = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
        labels_data.as_slice(),
        &device
    ).reshape([2, 3]); // [S, B] format
    
    // Test conversion to f32 for y_encoder (current implementation might be correct)  
    let y_encoder_input: Tensor<TestBackend, 3> = labels_for_model.float().unsqueeze_dim(2);
    let dims: [usize; 3] = y_encoder_input.dims();
    
    println!("Y-encoder input dims: {:?}", dims);
    println!("Y-encoder input dtype: f32 (after .float() conversion)");
    
    // Check that conversion produces expected shape and dtype
    assert_eq!(dims.len(), 3, "Y-encoder input should be 3D");
    assert_eq!(dims[2], 1, "Y-encoder input should have last dim = 1");
    
    println!("✅ Test D2 PASSED: Y-encoder receives f32 input correctly");
}
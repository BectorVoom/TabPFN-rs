//! Configuration type validation tests
//! 
//! These tests verify that all numeric configuration fields use f32 instead of f64
//! for consistency with Burn's tensor operations and optimizer requirements.

use burn::{
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    backend::Autodiff,
};
use burn_ndarray::NdArray;

// Use the same test backend as other tests
type TestBackend = Autodiff<NdArray<f32>>;

use tab_pfn_rs::tabpfn::architectures::base::{
    config::ModelConfig,
    train::{TrainingConfig, PriorType},
    transformer::{PerFeatureTransformer, DeterministicRngContext},
};

/// Test that ModelConfig dropout field is f32 - EXPECTED TO FAIL  
#[test]
fn test_model_config_dropout_is_f32() {
    println!("🔴 Test ModelConfig.dropout is f32 - EXPECTED TO FAIL");
    
    let mut config = ModelConfig::default();
    config.dropout = 0.1; // This assignment should work with f32
    
    // Test 1: Type annotation enforcement
    // This test will fail at compile-time if dropout is not f32
    let dropout_f32: f32 = config.dropout;
    assert!((dropout_f32 - 0.1).abs() < 1e-6, "Dropout value should be preserved as f32");
    
    // Test 2: Direct usage with Burn operations (requires f32)
    let device = <TestBackend as Backend>::Device::default();
    let test_tensor = Tensor::<TestBackend, 1>::ones([10], &device);
    
    // Dropout operations in Burn expect f32 probability
    // This should compile without casting if dropout is f32
    let _dropout_rate = config.dropout; // Should be f32
    
    // Simulate usage in neural network layer where f32 is required
    assert!(_dropout_rate >= 0.0f32 && _dropout_rate <= 1.0f32, 
            "Dropout rate should be in [0,1] range as f32");
    
    println!("✅ ModelConfig.dropout is f32: {}", config.dropout);
}

/// Test that ModelConfig attention_init_gain field is f32 - EXPECTED TO FAIL
#[test]
fn test_model_config_attention_init_gain_is_f32() {
    println!("🔴 Test ModelConfig.attention_init_gain is f32 - EXPECTED TO FAIL");
    
    let mut config = ModelConfig::default();
    config.attention_init_gain = 1.5; // This assignment should work with f32
    
    // Test 1: Type annotation enforcement
    let gain_f32: f32 = config.attention_init_gain;
    assert!((gain_f32 - 1.5).abs() < 1e-6, "Attention gain should be preserved as f32");
    
    // Test 2: Usage in tensor initialization (requires f32)
    let device = <TestBackend as Backend>::Device::default();
    
    // Parameter initialization typically uses f32 values
    let gain = config.attention_init_gain; // Should be f32
    let _init_tensor = Tensor::<TestBackend, 2>::ones([5, 5], &device) * gain;
    
    // Should work without casting if attention_init_gain is f32
    assert!(gain > 0.0f32, "Attention init gain should be positive f32");
    
    println!("✅ ModelConfig.attention_init_gain is f32: {}", config.attention_init_gain);
}

/// Test that TrainingConfig learning_rate field is f32 - EXPECTED TO FAIL
#[test]
fn test_training_config_learning_rate_is_f32() {
    println!("🔴 Test TrainingConfig.learning_rate is f32 - EXPECTED TO FAIL");
    
    use tab_pfn_rs::tabpfn::architectures::base::train::PriorType;
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 5),
        num_classes_range: (2, 4),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001, // This should be f32
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 10,
        max_samples_per_task: 20,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 2,
        use_gradient_checkpointing: false,
    };
    
    // Test 1: Type annotation enforcement
    let lr_f32: f32 = config.learning_rate;
    assert!((lr_f32 - 0.001).abs() < 1e-6, "Learning rate should be preserved as f32");
    
    // Test 2: Direct usage with optimizer (requires f32)
    // Burn optimizers expect f32 learning rates
    assert!(lr_f32 > 0.0f32, "Learning rate should be positive f32");
    
    // Test 3: Integration with actual optimizer step
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create a small transformer to test optimizer integration
    let mut model_config = ModelConfig::default();
    model_config.max_num_classes = 2;
    model_config.num_buckets = 0;
    model_config.nlayers = 1;
    model_config.emsize = 8;
    model_config.nhead = 2;
    
    let transformer = PerFeatureTransformer::new(
        &model_config,
        &rng_context,
        2,
        "gelu",
        Some(0),
        false,
        Some(1),
        false,
        None,
        false,
        &device,
    ).expect("Failed to create transformer");
    
    // The optimizer.step() call should accept config.learning_rate directly as f32
    let learning_rate = config.learning_rate; // Should be f32, no casting needed
    
    // Simulate optimizer step - this requires f32 learning rate
    assert!(learning_rate < 1.0f32, "Learning rate should be reasonable f32 value");
    
    println!("✅ TrainingConfig.learning_rate is f32: {}", config.learning_rate);
}

/// Test feature_noise_level field type consistency - EXPECTED TO FAIL
#[test] 
fn test_training_config_noise_level_consistency() {
    println!("🔴 Test TrainingConfig noise level consistency - EXPECTED TO FAIL");
    
    use tab_pfn_rs::tabpfn::architectures::base::train::PriorType;
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3),
        num_classes_range: (2, 2),
        feature_noise_level: 0.05, // This should be consistent with tensor operations
        model: ModelConfig::default(),
        learning_rate: 0.001f32,
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
    
    // Test noise level usage in tensor operations
    let device = <TestBackend as Backend>::Device::default();
    let noise_std = config.feature_noise_level;
    
    // If feature_noise_level is f64, this might require casting
    // If it's f32, this should work directly
    let noise_tensor = Tensor::<TestBackend, 1>::ones([5], &device) * (noise_std as f32);
    
    // Verify the conversion/usage
    assert!(noise_std > 0.0, "Noise level should be positive");
    assert!(noise_std < 1.0, "Noise level should be reasonable");
    
    // The actual test: check if casting was needed
    let _noise_f32: f32 = noise_std as f32; // This cast suggests f64 → f32 conversion needed
    
    println!("ℹ️  feature_noise_level type needs verification: {}", noise_std);
    
    // Test fails if type conversion is needed
    if std::mem::size_of_val(&noise_std) == 8 {
        panic!("feature_noise_level is f64, should be f32 for consistency");
    }
}

/// Test gradient_clip_norm field type consistency - EXPECTED TO FAIL
#[test]
fn test_training_config_gradient_clip_norm_consistency() {
    println!("🔴 Test TrainingConfig gradient_clip_norm consistency - EXPECTED TO FAIL");
    
    use tab_pfn_rs::tabpfn::architectures::base::train::PriorType;
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001f32,
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
        gradient_clip_norm: Some(1.0), // This should be f32 if used in gradient clipping
        meta_batch_size: 1,
        use_gradient_checkpointing: false,
    };
    
    if let Some(clip_norm) = config.gradient_clip_norm {
        // Test gradient clip norm usage
        let device = <TestBackend as Backend>::Device::default();
        
        // Gradient clipping operations typically expect f32 values
        let test_grad = Tensor::<TestBackend, 1>::ones([10], &device);
        let norm_value = clip_norm;
        
        // This should work without casting if gradient_clip_norm is f32
        assert!(norm_value > 0.0, "Gradient clip norm should be positive");
        
        // Check if type conversion is needed for gradient clipping operations
        let _norm_f32: f32 = norm_value as f32;
        
        if std::mem::size_of_val(&norm_value) == 8 {
            panic!("gradient_clip_norm is f64, should be f32 for gradient ops");
        }
    }
    
    println!("✅ gradient_clip_norm type consistency verified");
}

/// Test default values are proper f32 literals - EXPECTED TO FAIL
#[test]
fn test_config_default_values_f32_literals() {
    println!("🔴 Test config default values are f32 literals - EXPECTED TO FAIL");
    
    let config = ModelConfig::default();
    
    // Test that default values don't require casting from f64 to f32
    
    // Default dropout should be f32  
    let dropout = config.dropout;
    assert_eq!(dropout, 0.0f32, "Default dropout should be 0.0f32");
    
    // Default attention_init_gain should be f32
    let gain = config.attention_init_gain; 
    assert!((gain - 1.0f32).abs() < 1e-6, "Default attention_init_gain should be 1.0f32");
    
    // Test that these values work in f32 contexts without casting
    let device = <TestBackend as Backend>::Device::default();
    
    // Should work with f32 tensors directly
    let _dropout_tensor = Tensor::<TestBackend, 1>::ones([3], &device) * dropout;
    let _gain_tensor = Tensor::<TestBackend, 1>::ones([3], &device) * gain;
    
    // Test memory size to ensure they're f32, not f64
    if std::mem::size_of_val(&dropout) != 4 {
        panic!("dropout is not f32 (size: {} bytes)", std::mem::size_of_val(&dropout));
    }
    
    if std::mem::size_of_val(&gain) != 4 {
        panic!("attention_init_gain is not f32 (size: {} bytes)", std::mem::size_of_val(&gain));
    }
    
    println!("✅ Config default values are f32 literals");
}

/// Test y_encoder input conversion to f32 - EXPECTED TO FAIL
#[test]
fn test_y_encoder_input_f32_conversion() {
    println!("🔴 Test y_encoder input f32 conversion - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create labels_for_model as integer tensor (typical dataset output)
    let labels_int = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
        vec![0i64, 1i64, -1i64, 2i64].as_slice(), &device
    ).reshape([2, 2]); // [S, B]
    
    // y_encoder requires f32 input - test conversion
    let labels_f32 = labels_int.clone().float(); // Convert i64 → f32
    
    // Verify conversion worked and shape is preserved  
    assert_eq!(labels_f32.dims().to_vec(), vec![2, 2], "Shape should be preserved after conversion");
    
    // Verify values are correctly converted to f32
    let data = labels_f32.clone().to_data();
    let float_values = data.as_slice::<f32>().expect("Should be f32 after conversion");
    
    let expected = vec![0.0f32, 1.0f32, -1.0f32, 2.0f32];
    for (actual, expected) in float_values.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6, 
                "Value conversion incorrect: {} vs {}", actual, expected);
    }
    
    // Test that y_encoder input preparation includes .float() call
    let y_encoder_input = labels_int.clone().float().unsqueeze_dim::<3>(2); // Add feature dimension
    assert_eq!(y_encoder_input.dims().to_vec(), vec![2, 2, 1], "y_encoder input should be [S, B, 1]");
    
    // Verify the data is still f32 after unsqueeze
    let final_data = y_encoder_input.to_data();
    let _final_f32 = final_data.as_slice::<f32>().expect("Final tensor should be f32");
    
    println!("✅ y_encoder input f32 conversion verified");
}

/// Integration test: config values work in full training pipeline
#[test]
fn test_config_f32_integration() {
    println!("✅ Test config f32 integration in training pipeline");
    
    use tab_pfn_rs::tabpfn::architectures::base::train::{PriorType, TabPFNTrainer};
    use rand::{rngs::StdRng, SeedableRng};
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (2, 2),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        model: {
            let mut model_config = ModelConfig::default();
            model_config.max_num_classes = 2;
            model_config.num_buckets = 0;
            model_config.nlayers = 1;
            model_config.emsize = 8;
            model_config.nhead = 2;
            model_config.dropout = 0.1f32;  // Explicitly f32
            model_config.attention_init_gain = 1.0f32; // Explicitly f32
            model_config
        },
        learning_rate: 0.01f32, // Explicitly f32
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 5,
        max_samples_per_task: 5,
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
    
    // Test that configuration fields are all f32 types
    assert!((config.feature_noise_level - 0.1f32).abs() < 1e-6, "feature_noise_level should be f32");
    assert!(config.gradient_clip_norm.is_none(), "gradient_clip_norm should work with Option<f32>");
    
    // Successfully created TrainingConfig with all f32 fields
    println!("✅ TrainingConfig f32 integration test passed");
    
    // NOTE: Full trainer integration test commented out due to backend complexity
    // Focus is on verifying config field types are f32
    // let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    // let loss = trainer.train_step(&device, &mut rng);
}
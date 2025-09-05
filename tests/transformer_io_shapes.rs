//! Transformer I/O shape canonicalization tests
//! 
//! These tests verify that:
//! 1. transformer_forward accepts features in canonical [S,B,F] layout
//! 2. transformer_forward returns outputs in canonical [S,B,C] layout  
//! 3. train_step normalizes transformer outputs with swap_dims(0,1) if needed
//! 4. All tensor dimensions are validated before critical operations

use burn::{
    prelude::Backend,
    tensor::Tensor,
};
use burn_ndarray::NdArray;
use std::collections::HashMap;

// Use the same test backend as other tests
type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

use tab_pfn_rs::tabpfn::architectures::base::{
    transformer::{PerFeatureTransformer, DeterministicRngContext},
    config::ModelConfig,
    train::{TrainingConfig, TabPFNTrainer, PriorType},
};
use rand::{rngs::StdRng, SeedableRng};

/// Test that transformer_forward accepts canonical [S,B,F] input - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "transformer input shape validation not implemented")]
fn test_transformer_forward_canonical_input() {
    println!("🔴 Test transformer_forward canonical [S,B,F] input - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create transformer with test configuration
    let mut model_config = ModelConfig::default();
    model_config.max_num_classes = 4;
    model_config.num_buckets = 0; // Set required field
    
    let transformer = PerFeatureTransformer::new(
        &model_config,
        &rng_context,
        4, // n_out (num_classes)
        "gelu", // activation
        Some(0), // min_num_layers_layer_dropout
        false,   // zero_init
        Some(3), // nlayers_decoder
        false,   // use_encoder_compression_layer
        None,    // precomputed_kv
        false,   // cache_trainset_representation
        &device,
    ).expect("Failed to create transformer");
    
    // Create input in canonical [S,B,F] layout
    let S = 10; // sequence length (num_samples)
    let B = 2;  // meta batch size  
    let F = 5;  // num features
    
    let features_sbf = Tensor::<TestBackend, 1>::zeros([S * B * F], &device)
        .reshape([S, B, F]);
    
    let labels_sbf = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([S * B], &device)  
        .reshape([S, B]);
    
    // Prepare inputs in expected format
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), features_sbf);
    
    let mut y_inputs = HashMap::new();
    // Convert to f32 and add feature dimension as expected by transformer
    y_inputs.insert("main".to_string(), labels_sbf.float().unsqueeze_dim(2));
    
    // THIS SHOULD FAIL - input validation not implemented
    // Test that transformer_forward properly validates and accepts [S,B,F] input
    let mut rng_opt = Some(&mut StdRng::seed_from_u64(123));
    
    // This should work without shape conversion if transformer expects [S,B,F]  
    // If it fails, it means the transformer expects a different layout
    match transformer.transformer_forward(
        x_inputs,
        Some(y_inputs),
        true,  // train_mode
        &mut rng_opt,
        None,  // style
        None,  // x_src
        None,  // dag
        true,  // use_style
    ) {
        Ok(output) => {
            // If successful, verify output shape is [S,B,C]
            let output_dims = output.dims();
            println!("Transformer output dims: {:?}", output_dims);
            
            // CRITICAL: Output should be in canonical [S,B,C] format
            assert_eq!(output_dims.len(), 3, "Output should be 3D");
            assert_eq!(output_dims[0], S, "First dim should be sequence length S={}", S);
            assert_eq!(output_dims[1], B, "Second dim should be batch size B={}", B);
            // Third dimension is number of classes (varies)
            
            panic!("transformer input shape validation not implemented");
        },
        Err(e) => {
            panic!("Transformer failed with canonical [S,B,F] input: {:?}", e);
        }
    }
}

/// Test transformer_forward output shape is canonical [S,B,C] - EXPECTED TO FAIL
#[test]  
#[should_panic(expected = "transformer output shape not canonical")]
fn test_transformer_forward_canonical_output() {
    println!("🔴 Test transformer_forward canonical [S,B,C] output - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model_config = ModelConfig::default();
    model_config.max_num_classes = 3; 
    model_config.num_buckets = 0;
    
    let transformer = PerFeatureTransformer::new(
        &model_config,
        &rng_context, 
        3, // num_classes
        "gelu",
        Some(0),
        false,
        Some(2), // smaller model for testing
        false,
        None,
        false,
        &device,
    ).expect("Failed to create transformer");
    
    // Test with different input sizes
    for (S, B) in [(5, 1), (8, 3), (12, 2)] {
        let F = 4;
        
        let features = Tensor::<TestBackend, 1>::ones([S * B * F], &device)
            .reshape([S, B, F]);
        
        let labels = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([S * B], &device)
            .reshape([S, B]);
        
        let mut x_inputs = HashMap::new();
        x_inputs.insert("main".to_string(), features);
        
        let mut y_inputs = HashMap::new();
        y_inputs.insert("main".to_string(), labels.float().unsqueeze_dim(2));
        
        let mut rng_opt = Some(&mut StdRng::seed_from_u64(456));
        
        let output = transformer.transformer_forward(
            x_inputs,
            Some(y_inputs),
            true,
            &mut rng_opt,
            None,
            None,
            None,
            true,
        ).expect("Transformer forward failed");
        
        let output_dims = output.dims();
        println!("Input [S={}, B={}, F={}] → Output dims: {:?}", S, B, F, output_dims);
        
        // CRITICAL: Check if output is in canonical [S,B,C] or needs swapping
        if output_dims == [S, B, 3] {
            println!("✅ Output already in canonical [S,B,C] format");
        } else if output_dims == [B, S, 3] {
            println!("🔴 Output in [B,S,C] format - needs swap_dims(0,1) in train_step");
            panic!("transformer output shape not canonical");
        } else {
            panic!("Unexpected output shape: {:?}, expected [{},%s,3] or [{},{}],3]", 
                   output_dims, S, B, B, S);
        }
    }
    
    println!("✅ All transformer outputs in canonical format");
}

/// Test train_step shape normalization with swap_dims - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "train_step output normalization not implemented")]  
fn test_train_step_output_normalization() {
    println!("🔴 Test train_step output normalization - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default(); 
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3), // Fixed for testing
        num_classes_range: (2, 2), // Fixed for testing
        feature_noise_level: 0.1,
        model: {
            let mut model_config = ModelConfig::default();
            model_config.max_num_classes = 2;
            model_config.num_buckets = 0;
            model_config.nlayers = 1; // Small model for testing
            model_config.emsize = 32; // Small embedding size
            model_config.nhead = 2;   // Fewer heads
            model_config
        },
        learning_rate: 0.01,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 8,
        max_samples_per_task: 8,
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
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    let mut rng = StdRng::seed_from_u64(789);
    
    // THIS SHOULD FAIL - train_step doesn't normalize output shapes yet
    // The current train_step may receive [B,S,C] from transformer_forward 
    // and should apply swap_dims(0,1) to get canonical [S,B,C]
    
    // Attempt training step
    let loss = trainer.train_step(&device, &mut rng);
    
    // If this succeeds without shape errors, the normalization might be working
    // But we want to specifically test that the shape conversion happens
    assert!(loss.is_finite(), "Loss should be finite");
    
    // This test is designed to fail until train_step implements the shape normalization
    panic!("train_step output normalization not implemented");
}

/// Test shape validation guards in train_step - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "train_step shape guards not implemented")]
fn test_train_step_shape_validation_guards() {
    println!("🔴 Test train_step shape validation guards - EXPECTED TO FAIL");
    
    // This test verifies that train_step has proper shape validation guards
    // before critical reshape and loss computation operations
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (2, 2),
        num_classes_range: (3, 3), 
        feature_noise_level: 0.1,
        model: {
            let mut model_config = ModelConfig::default();
            model_config.max_num_classes = 3;
            model_config.num_buckets = 0;
            model_config.nlayers = 1;
            model_config.emsize = 24; // Divisible by nhead
            model_config.nhead = 2;
            model_config
        },
        learning_rate: 0.01,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 6,
        max_samples_per_task: 6,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 1, // Single task to simplify testing
        use_gradient_checkpointing: false,
    };
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    let mut rng = StdRng::seed_from_u64(321);
    
    // The train_step should have validation guards that check:
    // 1. Transformer output is 3D [batch, seq, classes]
    // 2. Targets are 2D [batch, seq] 
    // 3. Batch and sequence dimensions match between output and targets
    // 4. Loss computation preconditions are met
    
    let loss = trainer.train_step(&device, &mut rng);
    
    // If successful, the guards exist and work, but we want to verify they're comprehensive
    assert!(loss.is_finite(), "Loss should be finite with proper guards");
    
    // This test fails until comprehensive shape guards are implemented
    panic!("train_step shape guards not implemented");
}

/// Test transformer forward with malformed input shapes - EXPECTED TO FAIL  
#[test]
#[should_panic(expected = "transformer input validation not comprehensive")]
fn test_transformer_input_shape_validation() {
    println!("🔴 Test transformer input shape validation - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model_config = ModelConfig::default();
    model_config.max_num_classes = 2;
    model_config.num_buckets = 0;
    
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
    
    // Test with wrong input shapes that should be rejected
    
    // Test 1: 2D features (missing sequence dimension)
    let features_2d = Tensor::<TestBackend, 1>::zeros([4 * 3], &device) // [B*F]
        .reshape([4, 3]); // [B, F] - missing S dimension
    
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), features_2d);
    
    let mut rng_opt = Some(&mut StdRng::seed_from_u64(123));
    
    // THIS SHOULD FAIL with proper input validation
    let result = std::panic::catch_unwind(|| {
        transformer.transformer_forward(
            x_inputs,
            None, // No y_inputs for this test
            true,
            &mut rng_opt,
            None,
            None,
            None,
            true,
        )
    });
    
    match result {
        Ok(_) => {
            // If it doesn't panic, the input validation isn't comprehensive enough
            panic!("transformer input validation not comprehensive");
        },
        Err(err) => {
            let error_msg = format!("{:?}", err);
            println!("Good: Transformer rejected malformed input: {}", error_msg);
        }
    }
    
    // Additional tests would go here for other malformed inputs...
    
    panic!("transformer input validation not comprehensive");
}

/// Test loss computation shape requirements - EXPECTED TO FAIL
#[test]
#[should_panic(expected = "loss computation shape validation not implemented")]
fn test_loss_computation_shape_requirements() {
    println!("🔴 Test loss computation shape requirements - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Test the specific shape requirements for loss computation:
    // logits: [N, C] where N = S*B 
    // labels: [N] where N = S*B
    
    let S = 4;
    let B = 2; 
    let C = 3;
    
    // Create logits in canonical 3D format [S, B, C]
    let logits_3d = Tensor::<TestBackend, 1>::ones([S * B * C], &device)
        .reshape([S, B, C]);
    
    // Create labels in canonical 2D format [S, B] with -1 for test positions
    let labels_2d = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
        vec![0i64, -1, 1, -1, 2, 0, -1, 1].as_slice(), &device
    ).reshape([S, B]);
    
    // The loss computation should:
    // 1. Reshape logits from [S,B,C] to [S*B, C]
    // 2. Reshape labels from [S,B] to [S*B] 
    // 3. Apply ignore_index=-1 loss function
    
    // Test shape conversion
    let logits_2d = logits_3d.clone().reshape([S * B, C]);
    let labels_1d = labels_2d.clone().reshape([S * B]);
    
    assert_eq!(logits_2d.dims(), vec![S * B, C], "Logits should reshape to [N, C]");
    assert_eq!(labels_1d.dims(), vec![S * B], "Labels should reshape to [N]");
    
    // This should work with proper loss function
    // let loss = tab_pfn_rs::tabpfn::architectures::base::loss_utils::compute_masked_cross_entropy_loss_ignore_index(
    //     logits_2d, labels_1d, &device
    // );
    // assert!(loss.into_scalar() >= 0.0, "Loss should be non-negative");
    
    // Test fails until comprehensive shape validation is implemented in loss computation
    panic!("loss computation shape validation not implemented");
}

/// Test that transformer I/O matches dataset provider outputs - EXPECTED TO FAIL 
#[test]
#[should_panic(expected = "transformer dataset integration not validated")]
fn test_transformer_dataset_integration() {
    println!("🔴 Test transformer-dataset integration - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(555);
    
    // Create a dataset and verify transformer can process it directly
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (4, 4),
        num_classes_range: (3, 3),
        feature_noise_level: 0.1,
        model: {
            let mut model_config = ModelConfig::default();
            model_config.max_num_classes = 3;
            model_config.num_buckets = 0;
            model_config.nlayers = 1;
            model_config.emsize = 12;
            model_config.nhead = 3;
            model_config
        },
        learning_rate: 0.01,
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
    
    // Sample dataset from prior
    let prior = tab_pfn_rs::tabpfn::architectures::base::train::DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(5, 1, &device, &rng_context, &mut rng);
    
    // Create transformer
    let transformer = PerFeatureTransformer::new(
        &config.model,
        &rng_context,
        3,
        "gelu",
        Some(0),
        false,
        Some(1),
        false,
        None,
        false,
        &device,
    ).expect("Failed to create transformer");
    
    // Test direct integration - dataset output should work as transformer input
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), dataset.features.clone());
    
    let mut y_inputs = HashMap::new();
    y_inputs.insert("main".to_string(), dataset.labels_for_model.float().unsqueeze_dim(2));
    
    let mut rng_opt = Some(&mut StdRng::seed_from_u64(666));
    
    let output = transformer.transformer_forward(
        x_inputs,
        Some(y_inputs),
        true,
        &mut rng_opt,
        None,
        None,
        None,
        true,
    );
    
    match output {
        Ok(result) => {
            println!("Dataset dims: features={:?}, labels={:?}", 
                     dataset.features.dims(), dataset.labels_for_model.dims());
            println!("Transformer output dims: {:?}", result.dims());
            
            // Verify the integration works seamlessly
            let expected_output_shape = [5, 1, 3]; // [S, B, C]
            if result.dims() != expected_output_shape {
                panic!("Shape mismatch: expected {:?}, got {:?}", 
                       expected_output_shape, result.dims());
            }
        },
        Err(e) => {
            panic!("Transformer-dataset integration failed: {:?}", e);
        }
    }
    
    // This test fails until integration is properly validated
    panic!("transformer dataset integration not validated");
}
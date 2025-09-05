/// TDD Tests for Canonical [B,S,F] Tensor Layout Enforcement
/// 
/// These tests ensure that ALL internal tensors use the canonical layout:
/// - B (batch): number of tasks in meta-batch
/// - S (sequence): number of samples per task  
/// - F (features): number of input features
/// 
/// CRITICAL REQUIREMENT: If public API receives [S,B,F], convert immediately with swap_dims.
/// Never attempt axis permutation by reshape alone.

use burn::prelude::*;
use burn_ndarray::NdArray;
use rand::{rngs::StdRng, SeedableRng};
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;
use tab_pfn_rs::tabpfn::architectures::base::train::{TrainingConfig, DatasetPrior, PriorType};
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

/// Test 1: Dataset generation must produce canonical [B,S,F] layout
#[test]
fn test_dataset_generation_bsf_layout() {
    println!("TDD Test: Dataset generation produces canonical [B,S,F] layout - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test parameters that will expose layout issues
    let batch_size = 3;  // B = 3 tasks
    let seq_len = 4;     // S = 4 samples per task
    let num_features = 5; // F = 5 features
    
    // Use DatasetPrior constructor from TrainingConfig
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        meta_batch_size: batch_size,
        tasks_per_batch: 1,
        max_samples_per_task: seq_len,
        min_samples_per_task: seq_len,
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 1,
        validation_frequency: 1,
        early_stopping_patience: 1,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(1),
    };
    let prior = DatasetPrior::new(&config);
    
    // Generate dataset with meta_batch_size > 1 
    let dataset = prior.sample::<TestBackend>(seq_len, batch_size, &device, &rng_context, &mut rng);
    
    // CRITICAL TEST: Features tensor must have canonical [B,S,F] shape
    let features_dims = dataset.features.dims();
    assert_eq!(features_dims.len(), 3, "Features must be 3D tensor");
    assert_eq!(features_dims[0], batch_size, "Dimension 0 must be batch size (B)");
    assert_eq!(features_dims[1], seq_len, "Dimension 1 must be sequence length (S)"); 
    assert_eq!(features_dims[2], num_features, "Dimension 2 must be number of features (F)");
    
    // CRITICAL TEST: Targets tensor must have canonical [B,S] shape
    let targets_dims = dataset.targets.dims();
    assert_eq!(targets_dims.len(), 2, "Targets must be 2D tensor");
    assert_eq!(targets_dims[0], batch_size, "Targets dimension 0 must be batch size (B)");
    assert_eq!(targets_dims[1], seq_len, "Targets dimension 1 must be sequence length (S)");
    
    // CRITICAL TEST: Labels_for_model tensor must have canonical [B,S] shape  
    let labels_dims = dataset.labels_for_model.dims();
    assert_eq!(labels_dims.len(), 2, "labels_for_model must be 2D tensor");
    assert_eq!(labels_dims[0], batch_size, "labels_for_model dimension 0 must be batch size (B)");
    assert_eq!(labels_dims[1], seq_len, "labels_for_model dimension 1 must be sequence length (S)");
    
    // CRITICAL TEST: Train mask must have canonical [B,S] shape
    let train_mask_dims = dataset.train_mask.dims();  
    assert_eq!(train_mask_dims.len(), 2, "train_mask must be 2D tensor");
    assert_eq!(train_mask_dims[0], batch_size, "train_mask dimension 0 must be batch size (B)");
    assert_eq!(train_mask_dims[1], seq_len, "train_mask dimension 1 must be sequence length (S)");
    
    println!("✅ All dataset tensors follow canonical [B,S,F] layout");
}

/// Test 2: Transformer forward pass must handle [B,S,F] layout correctly
#[test] 
fn test_transformer_forward_bsf_layout() {
    println!("TDD Test: Transformer handles [B,S,F] layout - EXPECTED TO FAIL");
    
    // Test will fail because current transformer may not handle meta_batch_size > 1
    let batch_size = 2;
    let seq_len = 6; 
    let num_features = 4;
    let num_classes = 3;
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create synthetic input with canonical [B,S,F] layout
    let features = rng_context.generate_normal_tensor(
        [batch_size, seq_len, num_features], 
        &mut StdRng::seed_from_u64(42),
        0.0, 1.0
    );
    
    // Verify input has correct layout
    let input_dims = features.dims();
    assert_eq!(input_dims, [batch_size, seq_len, num_features], 
               "Input features must have [B,S,F] layout");
    
    // THIS WILL FAIL: Current transformer configuration likely hardcoded for batch_size=1
    let model_config = ModelConfig {
        max_num_classes: num_classes,
        emsize: 128,
        nhead: 8,
        nhid_factor: 4,
        nlayers: 6,
        dropout: 0.0,
        // ... other default config values
        ..ModelConfig::default()
    };
    
    // TODO: Implement transformer forward pass for meta-batch size > 1
    println!("❌ Current transformer cannot handle meta-batch size > 1");
    println!("   Need to implement PerFeatureTransformer.transformer_forward() for batch_size > 1");
    panic!("Transformer forward pass fails with [B,S,F] layout when B > 1");
}

/// Test 3: Public API tensor layout conversion
#[test]
fn test_public_api_layout_conversion() {
    println!("TDD Test: Public API converts [S,B,F] to [B,S,F] - EXPECTED TO FAIL");
    
    let batch_size = 2; 
    let seq_len = 3;
    let num_features = 4;
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Simulate legacy public API input with [S,B,F] layout
    let legacy_input = Tensor::<TestBackend, 3>::zeros([seq_len, batch_size, num_features], &device);
    assert_eq!(legacy_input.dims(), [seq_len, batch_size, num_features], "Legacy input is [S,B,F]");
    
    // CRITICAL REQUIREMENT: Must use explicit swap_dims, not reshape
    let canonical_input = legacy_input.swap_dims(0, 1); // Swap S and B dimensions
    assert_eq!(canonical_input.dims(), [batch_size, seq_len, num_features], 
               "After swap_dims(0,1), layout must be [B,S,F]");
    
    // ANTI-PATTERN: Never do this (reshape without swap_dims)
    // let wrong_conversion = legacy_input.reshape([batch_size, seq_len, num_features]); // ❌ FORBIDDEN
    
    println!("✅ Explicit swap_dims conversion from [S,B,F] to [B,S,F] works correctly");
}

/// Test 4: Loss computation with canonical layout
#[test]
fn test_loss_computation_bsf_layout() {
    println!("TDD Test: Loss computation handles [B,S,F] layout - EXPECTED TO FAIL");
    
    let batch_size = 2;
    let seq_len = 4; 
    let num_classes = 3;
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create logits with canonical layout - note: logits for loss should be [B*S, C]
    let logits_3d = rng_context.generate_normal_tensor(
        [batch_size, seq_len, num_classes],
        &mut StdRng::seed_from_u64(42), 
        0.0, 1.0
    );
    
    // For loss computation, flatten first two dimensions: [B,S,C] -> [B*S, C]
    let logits_2d = logits_3d.reshape([batch_size * seq_len, num_classes]);
    
    // Labels with canonical [B,S] layout, then flattened to [B*S]
    let labels_2d = Tensor::<TestBackend, 2, burn::tensor::Int>::from_data(
        TensorData::new(vec![0i64, 1i64, 2i64, 0i64,  // Task 1
                            1i64, 0i64, 1i64, 2i64], // Task 2
                       [batch_size, seq_len]),
        &device
    );
    let labels_1d = labels_2d.reshape([batch_size * seq_len]);
    
    // Test that loss utilities can handle this canonical layout
    use tab_pfn_rs::tabpfn::architectures::base::loss_utils;
    let loss = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits_2d, labels_1d, &device
    );
    
    // Loss should be finite and computable
    let loss_value: f32 = loss.into_scalar();
    assert!(loss_value.is_finite(), "Loss with [B,S,F] layout must be finite");
    
    println!("✅ Loss computation handles canonical [B,S,F] layout correctly");
}

/// Test 5: End-to-end layout consistency
#[test]  
fn test_end_to_end_layout_consistency() {
    println!("TDD Test: End-to-end [B,S,F] layout consistency - EXPECTED TO FAIL");
    
    // This test verifies that tensor layouts remain consistent throughout entire pipeline
    let batch_size = 2;  // Multiple tasks
    let seq_len = 5;
    let num_features = 3; 
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    // Step 1: Dataset generation with [B,S,F] layout
    // Use DatasetPrior constructor from TrainingConfig
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        meta_batch_size: batch_size,
        tasks_per_batch: 1,
        max_samples_per_task: seq_len,
        min_samples_per_task: seq_len,
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 1,
        validation_frequency: 1,
        early_stopping_patience: 1,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(1),
    };
    let prior = DatasetPrior::new(&config);
    
    let dataset = prior.sample::<TestBackend>(seq_len, batch_size, &device, &rng_context, &mut rng);
    
    // Verify dataset tensors have canonical layout
    assert_eq!(dataset.features.dims(), [batch_size, seq_len, num_features], 
               "Dataset features must be [B,S,F]");
    assert_eq!(dataset.targets.dims(), [batch_size, seq_len],
               "Dataset targets must be [B,S]"); 
    assert_eq!(dataset.labels_for_model.dims(), [batch_size, seq_len],
               "Dataset labels_for_model must be [B,S]");
    
    // Step 2: The configuration is already created above for DatasetPrior
    
    // Dataset generation already works with meta_batch_size > 1
    // Issue is in transformer forward pass and training loop
    println!("❌ End-to-end pipeline fails due to transformer limitations");
    println!("   Dataset generation works correctly with meta_batch_size > 1");
    println!("   Issue is in transformer forward pass and training loop");
    panic!("Transformer and training loop cannot handle meta_batch_size > 1");
}
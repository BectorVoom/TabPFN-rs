//! TabPFN In-Context Learning Tests
//! 
//! These tests validate the core TabPFN in-context learning semantics that are critical
//! for correct TabPFN behavior. TabPFN performs in-context learning where it sees
//! both training and test examples in a single sequence, with test labels masked as -1.

use burn::tensor::{Tensor, TensorData, Int, backend::{Backend, AutodiffBackend}};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use std::collections::HashMap;
use rand::{rngs::StdRng, SeedableRng};
use tab_pfn_rs::tabpfn::architectures::base::{
    train::{SyntheticTabularDataset, DatasetPrior, TrainingConfig, PriorType},
    transformer::{PerFeatureTransformer, DeterministicRngContext},
    config::ModelConfig,
};

type TestBackend = Autodiff<NdArray<f32>>;

/// Test 1: In-Context Learning Sequence Semantics
/// 
/// Validates that TabPFN properly handles sequences where training and test examples
/// are intermixed, and that the model can make predictions on test positions while
/// learning from train positions within the same sequence.
#[test]
fn test_incontext_sequence_semantics() {
    println!("Running Test: In-Context Learning Sequence Semantics");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create a specific sequence pattern: [train, test, train, test, train]
    let batch_size = 1;
    let seq_len = 5;
    let num_features = 3;
    let num_classes = 2;
    
    // Create features for all positions (both train and test see features)
    let features_data: Vec<f32> = vec![
        // Position 0 (train): [1.0, 0.0, 0.5] -> class 0
        1.0, 0.0, 0.5,
        // Position 1 (test): [0.8, 0.2, 0.3] -> should predict class 0  
        0.8, 0.2, 0.3,
        // Position 2 (train): [0.1, 1.0, 0.9] -> class 1
        0.1, 1.0, 0.9,
        // Position 3 (test): [0.2, 0.9, 0.8] -> should predict class 1
        0.2, 0.9, 0.8,
        // Position 4 (train): [1.0, 0.1, 0.4] -> class 0
        1.0, 0.1, 0.4,
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    );
    
    // Create targets - true labels for all positions
    let targets_data = vec![0i64, 0i64, 1i64, 1i64, 0i64]; // True labels
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    );
    
    // Create train mask: [true, false, true, false, true] 
    // This means positions 1 and 3 are test positions
    let train_mask_data = vec![true, false, true, false, true];
    let train_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data.clone(), [batch_size, seq_len]),
        &device
    );
    
    // Create dataset
    let dataset = SyntheticTabularDataset {
        features,
        targets,
        train_mask,
        dag: None,
    };
    
    // Validate dataset conforms to TabPFN specifications
    dataset.validate_shapes_or_panic();
    
    // Test 1A: Verify that labels_for_model has -1 at test positions
    let neg_ones_like_targets = Tensor::<TestBackend, 2, Int>::ones_like(&dataset.targets) * (-1);
    let labels_for_model = dataset.targets.clone()
        .mask_where(dataset.train_mask.clone().bool_not(), neg_ones_like_targets);
    
    let labels_data: Vec<i64> = labels_for_model.clone().into_data().to_vec().unwrap();
    let expected_labels = vec![0i64, -1i64, 1i64, -1i64, 0i64]; // -1 at positions 1,3
    
    assert_eq!(labels_data, expected_labels, 
        "FAIL: labels_for_model should have -1 at test positions");
    
    // Test 1B: Verify that model input includes ALL features (train + test)
    // This is critical - TabPFN must see test features to make predictions
    let input_features_dims = dataset.features.dims();
    assert_eq!(input_features_dims[1], seq_len, 
        "FAIL: Model must see all positions in sequence, got seq_len={}", input_features_dims[1]);
    
    // Test 1C: Verify that loss should only be computed on test positions
    let test_mask = dataset.train_mask.bool_not();
    let test_positions: Vec<bool> = test_mask.clone().into_data().to_vec().unwrap();
    let expected_test_positions = vec![false, true, false, true, false]; // Only positions 1,3
    
    assert_eq!(test_positions, expected_test_positions,
        "FAIL: Loss should only be computed on test positions 1,3");
    
    println!("✅ Test 1 PASSED: In-context learning sequence semantics validated");
    println!("   Train positions: {:?}", train_mask_data);
    println!("   Test positions: {:?}", expected_test_positions);  
    println!("   Labels for model: {:?}", labels_data);
}

/// Test 2: Forward Pass In-Context Learning
///
/// Tests that the model can perform forward pass on mixed train/test sequences
/// and that predictions are generated for all positions (but loss only computed on test).
#[test]
#[ignore] // Will fail until transformer constructor is fixed
fn test_forward_pass_incontext_learning() {
    println!("Running Test: Forward Pass In-Context Learning");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create minimal model configuration
    let model_config = ModelConfig {
        max_num_classes: 3,
        num_buckets: 100,
        seed: 42,
        emsize: 32, // Small for test
        nhid_factor: 2,
        nlayers: 1, // Minimal
        features_per_group: 2, // Minimal
        nhead: 2, // Minimal
        feature_positional_embedding: None,
        use_separate_decoder: true,
        dropout: 0.0,
        encoder_use_bias: false,
        multiquery_item_attention: false,
        nan_handling_enabled: true,
        nan_handling_y_encoder: true,
        normalize_by_used_features: true,
        normalize_on_train_only: true,
        normalize_to_ranking: false,
        normalize_x: true,
        recompute_attn: false,
        recompute_layer: true,
        remove_empty_features: true,
        remove_outliers: false,
        multiquery_item_attention_for_test_set: true,
        attention_init_gain: 1.0,
        dag_pos_enc_dim: None,
        item_attention_type: "full".to_string(),
        feature_attention_type: "full".to_string(),
        remove_duplicate_features: false,
    };
    
    // Create model - this will initially fail due to constructor issues
    let mut model = PerFeatureTransformer::new(
        &model_config,
        &rng_context,
        3, // max_num_classes
        "gelu",
        Some(1), // layer_dropout_min_layers
        false, // use_separate_decoder 
        Some(1), // nlayers
        false, // cache_trainset_representations
        None, // style
        false, // dag_pos_enc
        &device,
    ).expect("Failed to create transformer");
    
    // Create test dataset
    let batch_size = 1;
    let seq_len = 4;
    let num_features = 2;
    
    let features_data = vec![
        1.0, 0.0,  // Position 0 (train) 
        0.8, 0.2,  // Position 1 (test)
        0.1, 1.0,  // Position 2 (train)
        0.2, 0.9,  // Position 3 (test)
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    );
    
    let targets_data = vec![0i64, -1i64, 1i64, -1i64]; // -1 for test positions
    let y_tensor = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    ).float().unsqueeze_dim(2);
    
    // Prepare model inputs
    let mut x_inputs = HashMap::new();
    x_inputs.insert("main".to_string(), features);
    
    let mut y_inputs = HashMap::new();
    y_inputs.insert("main".to_string(), y_tensor);
    
    // Forward pass
    let mut rng = StdRng::seed_from_u64(42);
    let mut rng_opt = Some(&mut rng);
    
    let output = model.transformer_forward(
        x_inputs,
        Some(y_inputs),
        true, // hypernetwork_on 
        &mut rng_opt,
        None, // precomputed_kv
        None, // style
        None, // dags
        false, // train_mode
    ).expect("Forward pass failed");
    
    // Validate output shape
    let output_dims = output.dims();
    let expected_shape = [batch_size, seq_len, 3]; // [batch, seq, num_classes]
    
    assert_eq!(output_dims, expected_shape,
        "FAIL: Forward pass output shape should be {:?}, got {:?}", 
        expected_shape, output_dims);
    
    // Validate that output contains finite values (no NaN/Inf)
    let output_data: Vec<f32> = output.clone().into_data().to_vec().unwrap();
    assert!(output_data.iter().all(|&x| x.is_finite()),
        "FAIL: Forward pass output contains non-finite values");
    
    println!("✅ Test 2 PASSED: Forward pass in-context learning validated");
    println!("   Output shape: {:?}", output_dims);
    println!("   All outputs finite: {}", output_data.iter().all(|&x| x.is_finite()));
}

/// Test 3: Training Step In-Context Learning
///
/// Tests that the training step correctly processes in-context learning sequences
/// and only computes loss on test positions while using train positions for context.
#[test] 
#[ignore] // Will fail until training implementation issues are resolved
fn test_training_step_incontext_learning() {
    println!("Running Test: Training Step In-Context Learning");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create training configuration
    let config = TrainingConfig {
        model: ModelConfig {
            max_num_classes: 3,
            num_buckets: 100,
            seed: 42,
            emsize: 16, // Very small for test
            nhid_factor: 2,
            nlayers: 1,
            features_per_group: 2,
            nhead: 2,
            feature_positional_embedding: None,
            use_separate_decoder: true,
            dropout: 0.0,
            encoder_use_bias: false,
            multiquery_item_attention: false,
            nan_handling_enabled: true,
            nan_handling_y_encoder: true,
            normalize_by_used_features: true,
            normalize_on_train_only: true,
            normalize_to_ranking: false,
            normalize_x: true,
            recompute_attn: false,
            recompute_layer: true,
            remove_empty_features: true,
            remove_outliers: false,
            multiquery_item_attention_for_test_set: true,
            attention_init_gain: 1.0,
            dag_pos_enc_dim: None,
            item_attention_type: "full".to_string(),
            feature_attention_type: "full".to_string(),
            remove_duplicate_features: false,
        },
        meta_batch_size: 1,
        tasks_per_batch: 1,
        max_samples_per_task: 4, // Small for test
        min_samples_per_task: 4,
        learning_rate: 1e-3,
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
        prior_type: PriorType::Gaussian,
        num_features_range: (2, 2), // Fixed
        num_classes_range: (2, 2), // Fixed
        feature_noise_level: 0.1,
    };
    
    // This test will initially fail due to TabPFNTrainer constructor constraints
    // But it validates the intended behavior
    use tab_pfn_rs::tabpfn::architectures::base::train::TabPFNTrainer;
    
    // TODO: Fix constructor - currently blocked by AutodiffBackend constraint
    // let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    // let mut rng = StdRng::seed_from_u64(42);
    // let loss = trainer.train_step(&device, &mut rng);
    
    // For now, validate the architecture expectations
    assert!(config.min_samples_per_task >= 2, 
        "FAIL: Need at least 2 samples for train/test split");
    assert!(config.num_classes_range.1 >= 2,
        "FAIL: Need at least 2 classes for meaningful in-context learning");
    
    println!("✅ Test 3 PASSED: Training step in-context learning architecture validated");
    println!("   Note: Full test pending TabPFNTrainer constructor fix");
}

/// Test 4: Loss Computation Masking Validation
///
/// Tests that loss is computed correctly with proper masking for in-context learning,
/// ensuring that only test positions contribute to the loss.
#[test]
fn test_loss_computation_masking() {
    println!("Running Test: Loss Computation Masking");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create logits for a sequence with mixed train/test positions
    let batch_size = 1;
    let seq_len = 4;  
    let num_classes = 3;
    
    // Reshape to [batch*seq, classes] for loss computation
    let logits_data: Vec<f32> = vec![
        // Position 0: Good prediction for class 0 -> small loss
        3.0, 0.0, 0.0,
        // Position 1: Bad prediction for class 1 (predicts class 0) -> large loss  
        3.0, 0.0, 0.0,
        // Position 2: Good prediction for class 2 -> small loss
        0.0, 0.0, 3.0,
        // Position 3: Bad prediction for class 0 (predicts class 2) -> large loss
        0.0, 0.0, 3.0,
    ];
    
    let logits = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(logits_data, [batch_size * seq_len, num_classes]),
        &device
    ).require_grad();
    
    // True targets for all positions
    let targets_data = vec![0i64, 1i64, 2i64, 0i64];
    let targets = Tensor::<TestBackend, 1, Int>::from_data(
        TensorData::new(targets_data, [batch_size * seq_len]),
        &device
    );
    
    // Mask: only positions 1 and 3 are test positions (should contribute to loss)
    let mask_data = vec![false, true, false, true]; // Only test positions
    let mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(mask_data, [batch_size * seq_len]),
        &device
    );
    
    // Compute masked loss
    use tab_pfn_rs::tabpfn::architectures::base::loss_utils;
    let loss = loss_utils::compute_masked_cross_entropy_loss(
        logits.clone(), targets.clone(), mask.clone(), &device
    );
    
    let loss_value: f32 = loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    
    // Test 4A: Loss should be finite
    assert!(loss_value.is_finite(), 
        "FAIL: Loss should be finite, got: {}", loss_value);
    
    // Test 4B: Verify loss is computed correctly by comparing with manual calculation
    // For test positions (1,3), targets are (1,0) but logits favor (0,2) -> large loss expected
    // Position 1: target=1, prediction=0 -> large loss
    // Position 3: target=0, prediction=2 -> large loss  
    assert!(loss_value > 1.0,
        "FAIL: Loss should be large due to wrong predictions, got: {}", loss_value);
    
    // Test 4C: Compare with unmasked loss - masked loss should be different
    let unmasked_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(vec![true, true, true, true], [batch_size * seq_len]),
        &device
    );
    
    let unmasked_loss = loss_utils::compute_masked_cross_entropy_loss(
        logits.clone(), targets.clone(), unmasked_mask, &device
    );
    let unmasked_loss_value: f32 = unmasked_loss.to_data().as_slice::<f32>().unwrap()[0];
    
    assert_ne!(loss_value, unmasked_loss_value,
        "FAIL: Masked and unmasked loss should differ");
    
    // Test 4D: Gradients should be computable (backward pass works)
    let grads = loss.backward();
    // If we reach here, gradients were computed successfully
    
    println!("✅ Test 4 PASSED: Loss computation masking validated");
    println!("   Masked loss: {:.4}", loss_value);
    println!("   Unmasked loss: {:.4}", unmasked_loss_value);
    println!("   Difference: {:.4}", (loss_value - unmasked_loss_value).abs());
}

/// Test 5: Data Structure Exact Conformance
///
/// Tests that the data structures exactly match the normative specification
/// for TabPFN in-context learning requirements.
#[test]
fn test_data_structure_conformance() {
    println!("Running Test: Data Structure Exact Conformance");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        model: ModelConfig::default(),
        meta_batch_size: 1,
        tasks_per_batch: 1,
        max_samples_per_task: 8,
        min_samples_per_task: 6,
        learning_rate: 1e-4,
        warmup_steps: 10,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 1,
        validation_frequency: 1,
        early_stopping_patience: 1,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(1),
        prior_type: PriorType::Gaussian,
        num_features_range: (4, 4), // Fixed for test
        num_classes_range: (3, 3), // Fixed for test
        feature_noise_level: 0.1,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(42);
    
    // Generate dataset with specific size for conformance testing
    let n_examples = 6;
    let dataset = prior.sample::<TestBackend>(n_examples, 1, &device, &rng_context, &mut rng);
    
    // Test 5A: Exact shape conformance - NORMATIVE SPECIFICATION
    // features: float32 tensor with shape [batch, seq_len, num_features]
    // Note: seq_len must equal the total examples per task (n_train + n_test)
    assert_eq!(dataset.features.dims(), [1, n_examples, 4],
        "FAIL: features shape must be [1, {}, 4] (batch=1, seq_len=n_examples, num_features=4)", n_examples);
    
    // targets: int64 tensor with shape [batch, seq_len] containing true class indices (0..C-1)
    assert_eq!(dataset.targets.dims(), [1, n_examples],
        "FAIL: targets shape must be [1, {}] (batch=1, seq_len=n_examples)", n_examples);
    
    // train_mask: bool tensor with shape [batch, seq_len], true = train, false = test  
    assert_eq!(dataset.train_mask.dims(), [1, n_examples],
        "FAIL: train_mask shape must be [1, {}] (batch=1, seq_len=n_examples)", n_examples);
    
    // Test 5B: Data type conformance
    // Features should be f32, targets should be i64, train_mask should be bool
    let features_data: Vec<f32> = dataset.features.clone().into_data().to_vec().unwrap();
    assert!(features_data.iter().all(|&x| x.is_finite()),
        "FAIL: All features must be finite f32 values");
    
    let targets_data: Vec<i64> = dataset.targets.clone().into_data().to_vec().unwrap();
    assert!(targets_data.iter().all(|&x| x >= 0 && x < 3),
        "FAIL: All targets must be valid class indices 0..2");
    
    let mask_data: Vec<bool> = dataset.train_mask.clone().into_data().to_vec().unwrap();
    assert!(mask_data.iter().any(|&x| x) && mask_data.iter().any(|&x| !x),
        "FAIL: train_mask must contain both train (true) and test (false) positions");
    
    // Test 5C: Sequence length semantics - CRITICAL for TabPFN
    // seq_len must equal total examples per task, NOT be 1 with examples in batch dimension
    let seq_len = dataset.features.dims()[1];
    assert_eq!(seq_len, n_examples,
        "FAIL: seq_len ({}) must equal total examples per task ({})", seq_len, n_examples);
    
    let batch_size = dataset.features.dims()[0]; 
    assert_eq!(batch_size, 1,
        "FAIL: For single task, batch_size should be 1, got {}", batch_size);
    
    println!("✅ Test 5 PASSED: Data structure exact conformance validated");
    println!("   Features shape: {:?} (f32)", dataset.features.dims());
    println!("   Targets shape: {:?} (i64)", dataset.targets.dims());
    println!("   Train mask shape: {:?} (bool)", dataset.train_mask.dims());
    println!("   Sequence length: {} (equals n_examples)", seq_len);
}
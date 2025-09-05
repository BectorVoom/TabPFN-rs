/// Shape Specification Tests
/// 
/// Test suite to verify that DatasetPrior::sample() returns tensors with canonical [S,B,F] shapes
/// as specified in the TabPFN tensor specification. This enforces the mandatory shape requirements:
/// - Features: [S, B, F] (sequence length, meta-batch size, features)
/// - Targets: [S, B] (sequence length, meta-batch size) 
/// - Train mask: [S, B] (sequence length, meta-batch size)
/// - Labels for model: [S, B] (sequence length, meta-batch size)

use burn::tensor::backend::Backend;
use burn_ndarray::{NdArray};
use rand::{rngs::StdRng, SeedableRng};

use tab_pfn_rs::{
    tabpfn::architectures::base::{
        config::ModelConfig,
        transformer::DeterministicRngContext,
        train::{DatasetPrior, TrainingConfig, PriorType},
    }
};

type TestBackend = NdArray<f32>;

/// Helper function to create a complete TrainingConfig for testing
fn create_test_training_config(
    num_features_range: (usize, usize),
    num_classes_range: (usize, usize),
    meta_batch_size: usize,
) -> TrainingConfig {
    TrainingConfig {
        model: ModelConfig {
            max_num_classes: 10,
            num_buckets: 1000,
            emsize: 512,
            features_per_group: 1,
            nhead: 8,
            remove_duplicate_features: false,
            dropout: 0.1,
            encoder_use_bias: true,
            feature_positional_embedding: None,
            multiquery_item_attention: false,
            nan_handling_enabled: true,
            nan_handling_y_encoder: true,
            nhid_factor: 4,
            nlayers: 6,
            normalize_by_used_features: true,
            normalize_on_train_only: true,
            normalize_to_ranking: false,
            normalize_x: true,
            recompute_attn: false,
            recompute_layer: true,
            remove_empty_features: true,
            remove_outliers: false,
            use_separate_decoder: false,
            multiquery_item_attention_for_test_set: true,
            attention_init_gain: 1.0,
            dag_pos_enc_dim: None,
            item_attention_type: "full".to_string(),
            feature_attention_type: "full".to_string(),
            seed: 0,
        },
        meta_batch_size,
        tasks_per_batch: 1,
        max_samples_per_task: 1000,
        min_samples_per_task: 10,
        learning_rate: 0.001,
        warmup_steps: 1000,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: Some(1.0),
        num_epochs: 1,
        checkpoint_frequency: 1000,
        validation_frequency: 100,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: None,
        prior_type: PriorType::Gaussian,
        num_features_range,
        num_classes_range,
        feature_noise_level: 0.1,
    }
}

/// Test that DatasetPrior::sample returns tensors with correct canonical shapes for small configuration
#[test]
fn test_dataset_prior_sample_shapes_small() {
    println!("Testing DatasetPrior::sample canonical shapes - Small config (S=5, B=2, F=3)");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(123);
    
    // Create dataset prior with known configuration
    let config = create_test_training_config(
        (3, 3), // Fixed to F=3
        (2, 4), // Variable classes
        2,      // B=2
    );
    
    let prior = DatasetPrior::new(&config);
    
    // Sample dataset with specific dimensions
    let seq_len = 5;      // S=5
    let meta_batch = 2;   // B=2  
    let expected_features = 3; // F=3 (from config)
    
    let dataset = prior.sample::<TestBackend>(seq_len, meta_batch, &device, &rng_context, &mut rng);
    
    // CRITICAL: Test canonical tensor shapes [S,B,F], [S,B], [S,B], [S,B]
    assert_eq!(
        dataset.features.dims(), 
        [seq_len, meta_batch, expected_features],
        "Features shape mismatch: expected [S={}, B={}, F={}], got {:?}. Features must be in canonical [S,B,F] format.",
        seq_len, meta_batch, expected_features, dataset.features.dims()
    );
    
    assert_eq!(
        dataset.targets.dims(),
        [seq_len, meta_batch], 
        "Targets shape mismatch: expected [S={}, B={}], got {:?}. Targets must be in canonical [S,B] format.",
        seq_len, meta_batch, dataset.targets.dims()
    );
    
    assert_eq!(
        dataset.train_mask.dims(),
        [seq_len, meta_batch],
        "Train mask shape mismatch: expected [S={}, B={}], got {:?}. Train mask must be in canonical [S,B] format.", 
        seq_len, meta_batch, dataset.train_mask.dims()
    );
    
    assert_eq!(
        dataset.labels_for_model.dims(),
        [seq_len, meta_batch],
        "Labels for model shape mismatch: expected [S={}, B={}], got {:?}. Labels for model must be in canonical [S,B] format.",
        seq_len, meta_batch, dataset.labels_for_model.dims()
    );
    
    println!("✅ Small config test PASSED: All tensors have canonical shapes");
}

/// Test that DatasetPrior::sample returns tensors with correct canonical shapes for medium configuration
#[test] 
fn test_dataset_prior_sample_shapes_medium() {
    println!("Testing DatasetPrior::sample canonical shapes - Medium config (S=10, B=4, F=5)");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(999, device.clone());
    let mut rng = StdRng::seed_from_u64(456);
    
    // Create dataset prior with medium configuration
    let config = create_test_training_config(
        (5, 5), // Fixed to F=5
        (3, 3), // Fixed to C=3
        4,      // B=4
    );
    
    let prior = DatasetPrior::new(&config);
    
    // Sample dataset with medium dimensions
    let seq_len = 10;     // S=10
    let meta_batch = 4;   // B=4
    let expected_features = 5; // F=5 (from config)
    
    let dataset = prior.sample::<TestBackend>(seq_len, meta_batch, &device, &rng_context, &mut rng);
    
    // Test canonical tensor shapes for medium config
    assert_eq!(
        dataset.features.dims(), 
        [seq_len, meta_batch, expected_features],
        "Medium config features shape mismatch: expected [S={}, B={}, F={}], got {:?}",
        seq_len, meta_batch, expected_features, dataset.features.dims()
    );
    
    assert_eq!(
        dataset.targets.dims(),
        [seq_len, meta_batch],
        "Medium config targets shape mismatch: expected [S={}, B={}], got {:?}",
        seq_len, meta_batch, dataset.targets.dims()
    );
    
    assert_eq!(
        dataset.train_mask.dims(),
        [seq_len, meta_batch],
        "Medium config train mask shape mismatch: expected [S={}, B={}], got {:?}",
        seq_len, meta_batch, dataset.train_mask.dims()
    );
    
    assert_eq!(
        dataset.labels_for_model.dims(),
        [seq_len, meta_batch],
        "Medium config labels for model shape mismatch: expected [S={}, B={}], got {:?}",
        seq_len, meta_batch, dataset.labels_for_model.dims()
    );
    
    println!("✅ Medium config test PASSED: All tensors have canonical shapes");
}

/// Test that DatasetPrior::sample returns tensors with correct canonical shapes for large configuration
#[test]
fn test_dataset_prior_sample_shapes_large() {
    println!("Testing DatasetPrior::sample canonical shapes - Large config (S=100, B=8, F=10)");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(777, device.clone());
    let mut rng = StdRng::seed_from_u64(789);
    
    // Create dataset prior with large configuration
    let config = create_test_training_config(
        (10, 10), // Fixed to F=10
        (5, 5),   // Fixed to C=5
        8,        // B=8
    );
    
    let prior = DatasetPrior::new(&config);
    
    // Sample dataset with large dimensions 
    let seq_len = 100;    // S=100
    let meta_batch = 8;   // B=8
    let expected_features = 10; // F=10 (from config)
    
    let dataset = prior.sample::<TestBackend>(seq_len, meta_batch, &device, &rng_context, &mut rng);
    
    // Test canonical tensor shapes for large config
    assert_eq!(
        dataset.features.dims(),
        [seq_len, meta_batch, expected_features],
        "Large config features shape mismatch: expected [S={}, B={}, F={}], got {:?}",
        seq_len, meta_batch, expected_features, dataset.features.dims()
    );
    
    assert_eq!(
        dataset.targets.dims(),
        [seq_len, meta_batch],
        "Large config targets shape mismatch: expected [S={}, B={}], got {:?}",
        seq_len, meta_batch, dataset.targets.dims()
    );
    
    assert_eq!(
        dataset.train_mask.dims(),
        [seq_len, meta_batch],
        "Large config train mask shape mismatch: expected [S={}, B={}], got {:?}", 
        seq_len, meta_batch, dataset.train_mask.dims()
    );
    
    assert_eq!(
        dataset.labels_for_model.dims(),
        [seq_len, meta_batch],
        "Large config labels for model shape mismatch: expected [S={}, B={}], got {:?}",
        seq_len, meta_batch, dataset.labels_for_model.dims()
    );
    
    println!("✅ Large config test PASSED: All tensors have canonical shapes");
}

/// Test that DatasetPrior::sample returns tensors with correct data types
#[test]
fn test_dataset_prior_sample_data_types() {
    println!("Testing DatasetPrior::sample data types");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(111, device.clone());
    let mut rng = StdRng::seed_from_u64(222);
    
    let config = create_test_training_config(
        (3, 3), // F=3
        (2, 2), // C=2
        2,      // B=2
    );
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(5, 2, &device, &rng_context, &mut rng);
    
    // Test data types by attempting to extract data
    // Features should be f32 (float type)
    let features_data = dataset.features.to_data();
    assert!(features_data.as_slice::<f32>().is_ok(), "Features must be f32 type");
    
    // Targets should be i64 (integer type for class indices)  
    let targets_data = dataset.targets.to_data();
    assert!(targets_data.as_slice::<i64>().is_ok(), "Targets must be i64 type for class indices");
    
    // Train mask should be bool type
    let mask_data = dataset.train_mask.to_data();
    assert!(mask_data.as_slice::<bool>().is_ok(), "Train mask must be bool type");
    
    // Labels for model should be i64 (integer type with -1 sentinel)
    let labels_data = dataset.labels_for_model.to_data(); 
    assert!(labels_data.as_slice::<i64>().is_ok(), "Labels for model must be i64 type");
    
    println!("✅ Data types test PASSED: All tensors have correct data types");
}

/// Test that different sequence lengths and batch sizes produce correct shapes
#[test]
fn test_dataset_prior_sample_variable_dimensions() {
    println!("Testing DatasetPrior::sample with variable S and B dimensions");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(333, device.clone());
    let mut rng = StdRng::seed_from_u64(444);
    
    let config = create_test_training_config(
        (4, 4), // F=4
        (3, 3), // C=3
        1,      // Will be overridden by sample() parameter
    );
    
    let prior = DatasetPrior::new(&config);
    
    // Test various combinations of S and B
    let test_cases = [
        (10, 1, 4),   // Small batch: S=10, B=1, F=4
        (50, 6, 4),   // Medium: S=50, B=6, F=4
        (20, 10, 4),  // High batch: S=20, B=10, F=4
    ];
    
    for (seq_len, meta_batch, expected_features) in test_cases {
        println!("  Testing S={}, B={}, F={}", seq_len, meta_batch, expected_features);
        
        let dataset = prior.sample::<TestBackend>(seq_len, meta_batch, &device, &rng_context, &mut rng);
        
        assert_eq!(
            dataset.features.dims(),
            [seq_len, meta_batch, expected_features],
            "Variable dimensions test failed for S={}, B={}, F={}: features shape {:?}",
            seq_len, meta_batch, expected_features, dataset.features.dims()
        );
        
        assert_eq!(
            dataset.targets.dims(),
            [seq_len, meta_batch],
            "Variable dimensions test failed for S={}, B={}: targets shape {:?}",
            seq_len, meta_batch, dataset.targets.dims()
        );
        
        // Quick validation that tensors contain reasonable values
        let targets_data = dataset.targets.to_data();
        if let Ok(targets_slice) = targets_data.as_slice::<i64>() {
            // All targets should be non-negative class indices
            assert!(targets_slice.iter().all(|&x| x >= 0), 
                "All target values must be non-negative class indices for S={}, B={}", seq_len, meta_batch);
        }
    }
    
    println!("✅ Variable dimensions test PASSED: All configurations produce canonical shapes");
}
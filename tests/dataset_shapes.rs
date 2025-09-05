use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    tensor::{Tensor, TensorData},
};
use tab_pfn_rs::tabpfn::architectures::base::{
    train::{DatasetPrior, SyntheticTabularDataset, TrainingConfig, PriorType},
    transformer::DeterministicRngContext,
    config::ModelConfig,
};
use rand::{rngs::StdRng, SeedableRng};

type TestBackend = Autodiff<NdArray>;

/// Test comprehensive dataset shape validation according to TabPFN specifications
/// 
/// This module tests the canonical tensor format requirements:
/// - Features: [S, B, F] where S ≥ 1, B ≥ 1, F ≥ 1
/// - Targets: [S, B] with values in [0, C)
/// - Train mask: [S, B] with at least one true and one false per task
/// - Labels for model: [S, B] with train labels and -1 for test positions

#[test]
fn test_dataset_canonical_shapes() {
    println!("🧪 Testing dataset canonical shape requirements");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create test configuration
    let mut config = TrainingConfig {
        meta_batch_size: 4,
        num_features_range: (5, 10),
        num_classes_range: (2, 5),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(123);
    
    // Test various dataset sizes
    let test_cases = vec![
        (10, 2, "minimal case"),     // S=10, B=2
        (100, 4, "standard case"),   // S=100, B=4  
        (50, 1, "single task"),      // S=50, B=1
        (2, 8, "minimal samples"),   // S=2, B=8 (minimum for train/test split)
    ];
    
    for (seq_len, batch_size, description) in test_cases {
        println!("  Testing {}: S={}, B={}", description, seq_len, batch_size);
        config.meta_batch_size = batch_size;
        let prior = DatasetPrior::new(&config);
        
        let dataset = prior.sample(seq_len, batch_size, &device, &rng_context, &mut rng);
        
        // Validate canonical shapes
        let features_dims = dataset.features.dims();
        let targets_dims = dataset.targets.dims();
        let train_mask_dims = dataset.train_mask.dims();
        let labels_dims = dataset.labels_for_model.dims();
        
        // REQUIREMENT: Features must be [S, B, F]
        assert_eq!(features_dims.len(), 3, "Features must be 3D [S,B,F]");
        assert_eq!(features_dims[0], seq_len, "Features sequence dimension must match S");
        assert_eq!(features_dims[1], batch_size, "Features batch dimension must match B");
        assert!(features_dims[2] >= 1, "Feature dimension F must be positive");
        
        // REQUIREMENT: Targets must be [S, B]
        assert_eq!(targets_dims.len(), 2, "Targets must be 2D [S,B]");
        assert_eq!(targets_dims, [seq_len, batch_size], "Targets shape must be [S,B]");
        
        // REQUIREMENT: Train mask must be [S, B]
        assert_eq!(train_mask_dims.len(), 2, "Train mask must be 2D [S,B]");
        assert_eq!(train_mask_dims, [seq_len, batch_size], "Train mask shape must be [S,B]");
        
        // REQUIREMENT: Labels for model must be [S, B]
        assert_eq!(labels_dims.len(), 2, "Labels for model must be 2D [S,B]");
        assert_eq!(labels_dims, [seq_len, batch_size], "Labels for model shape must be [S,B]");
        
        println!("    ✓ Shape validation passed: features {:?}", features_dims);
    }
}

#[test]
fn test_per_task_train_test_distribution() {
    println!("🧪 Testing per-task train/test distribution requirements");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 6,
        num_features_range: (3, 3), // Fixed for reproducibility
        num_classes_range: (2, 2),  // Fixed for reproducibility
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(456);
    
    // Test multiple datasets to ensure consistent train/test distribution
    for attempt in 0..5 {
        let dataset = prior.sample(20, 6, &device, &rng_context, &mut rng);
        
        // CRITICAL REQUIREMENT: Each task must have both training and test samples
        for task_idx in 0..6 {
            let task_mask = dataset.train_mask.clone().select(1, task_idx);
            let mask_data = task_mask.to_data();
            let mask_slice = mask_data.as_slice::<bool>().unwrap();
            
            let train_count = mask_slice.iter().filter(|&&x| x).count();
            let test_count = mask_slice.iter().filter(|&&x| !x).count();
            
            assert!(train_count > 0, 
                   "Attempt {}, Task {}: Must have at least one training sample, got {}", 
                   attempt, task_idx, train_count);
            assert!(test_count > 0, 
                   "Attempt {}, Task {}: Must have at least one test sample, got {}", 
                   attempt, task_idx, test_count);
            
            println!("    Task {}: {} train, {} test samples", task_idx, train_count, test_count);
        }
    }
    
    println!("✓ Per-task train/test distribution validation passed");
}

#[test]
fn test_labels_for_model_construction() {
    println!("🧪 Testing labels_for_model construction with -1 sentinel values");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 3,
        num_features_range: (4, 4), // Fixed
        num_classes_range: (3, 3),  // Fixed
        feature_noise_level: 0.0,   // No noise for deterministic testing
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(789);
    
    let dataset = prior.sample(10, 3, &device, &rng_context, &mut rng);
    
    // Extract tensor data for detailed validation
    let targets_data = dataset.targets.to_data();
    let train_mask_data = dataset.train_mask.to_data();
    let labels_data = dataset.labels_for_model.to_data();
    
    let targets_slice = targets_data.as_slice::<i64>().unwrap();
    let mask_slice = train_mask_data.as_slice::<bool>().unwrap();
    let labels_slice = labels_data.as_slice::<i64>().unwrap();
    
    // REQUIREMENT: labels_for_model = targets.mask_where(train_mask.bool_not(), -1)
    for (i, ((&target, &is_train), &label)) in targets_slice.iter()
        .zip(mask_slice.iter())
        .zip(labels_slice.iter())
        .enumerate() 
    {
        if is_train {
            assert_eq!(label, target, 
                      "Position {}: Training position should have original target {} but got {}", 
                      i, target, label);
        } else {
            assert_eq!(label, -1, 
                      "Position {}: Test position should have -1 sentinel but got {}", 
                      i, label);
        }
    }
    
    println!("✓ Labels for model construction validation passed");
}

#[test]
fn test_target_value_ranges() {
    println!("🧪 Testing target value ranges within [0, C)");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 4,
        num_features_range: (5, 5),
        num_classes_range: (4, 4), // Fixed at 4 classes
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(999);
    
    let dataset = prior.sample(50, 4, &device, &rng_context, &mut rng);
    
    // Validate target values are in valid range [0, 4)
    let targets_data = dataset.targets.to_data();
    let targets_slice = targets_data.as_slice::<i64>().unwrap();
    
    let mut class_counts = vec![0; 4];
    for &target in targets_slice {
        assert!(target >= 0, "Target {} must be non-negative", target);
        assert!(target < 4, "Target {} must be less than num_classes (4)", target);
        class_counts[target as usize] += 1;
    }
    
    // Verify that multiple classes are represented (not all same class)
    let non_zero_classes = class_counts.iter().filter(|&&count| count > 0).count();
    assert!(non_zero_classes > 1, 
           "Dataset should contain multiple classes, got distribution: {:?}", 
           class_counts);
    
    println!("✓ Target value range validation passed. Class distribution: {:?}", class_counts);
}

#[test]
fn test_dataset_validation_method() {
    println!("🧪 Testing SyntheticTabularDataset::validate_shapes_or_panic() method");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 2,
        num_features_range: (3, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(1111);
    
    // Generate valid dataset
    let dataset = prior.sample(30, 2, &device, &rng_context, &mut rng);
    
    // This should pass without panicking
    dataset.validate_shapes_or_panic();
    println!("✓ Valid dataset passed validation");
    
    // Test validate_shapes method (non-panicking version)
    match dataset.validate_shapes() {
        Ok(()) => println!("✓ validate_shapes() returned Ok for valid dataset"),
        Err(msg) => panic!("Valid dataset should not fail validation: {}", msg),
    }
}

#[test]
fn test_edge_case_minimum_dimensions() {
    println!("🧪 Testing edge cases with minimum valid dimensions");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Edge case: minimum sequence length (2 samples for train/test split)
    let config = TrainingConfig {
        meta_batch_size: 1,
        tasks_per_batch: 1,
        max_samples_per_task: 100,
        min_samples_per_task: 10,
        num_features_range: (1, 1), // Minimum features
        num_classes_range: (2, 2),  // Minimum meaningful classes
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 50,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        layer_dropout_min_layers: Some(2),
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(1234);
    
    // This should work with minimal dimensions
    let dataset = prior.sample(2, 1, &device, &rng_context, &mut rng);
    dataset.validate_shapes_or_panic();
    
    assert_eq!(dataset.features.dims(), [2, 1, 1], "Minimal features shape");
    assert_eq!(dataset.targets.dims(), [2, 1], "Minimal targets shape");
    
    println!("✓ Minimum dimension edge case passed");
}

#[test]
fn test_larger_dataset_scaling() {
    println!("🧪 Testing larger dataset scaling");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Larger dataset to test scaling
    let config = TrainingConfig {
        meta_batch_size: 16, // Larger batch
        tasks_per_batch: 1,
        max_samples_per_task: 100,
        min_samples_per_task: 10,
        num_features_range: (20, 25),
        num_classes_range: (5, 8),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 50,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        layer_dropout_min_layers: Some(2),
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(5678);
    
    let dataset = prior.sample(200, 16, &device, &rng_context, &mut rng);
    dataset.validate_shapes_or_panic();
    
    // Verify all tasks have proper train/test distribution
    for task_idx in 0..16 {
        let task_mask = dataset.train_mask.clone().slice([0..dataset.train_mask.dims()[0], task_idx..task_idx+1]).squeeze(1);
        let has_train = task_mask.clone().any().into_scalar();
        let has_test = task_mask.clone().bool_not().any().into_scalar();
        assert!(has_train && has_test, "Task {} missing train or test samples", task_idx);
    }
    
    println!("✓ Large dataset scaling test passed: shape {:?}", dataset.features.dims());
}

#[test]
#[should_panic(expected = "PARAMETER ERROR: num_samples must be ≥ 2")]
fn test_invalid_parameter_num_samples_too_small() {
    println!("🧪 Testing parameter validation: num_samples too small");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 2,
        tasks_per_batch: 1,
        max_samples_per_task: 100,
        min_samples_per_task: 10,
        num_features_range: (3, 3),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 50,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        layer_dropout_min_layers: Some(2),
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(999);
    
    // This should panic due to num_samples = 1 (too small for train/test split)
    prior.sample(1, 2, &device, &rng_context, &mut rng);
}

#[test]
#[should_panic(expected = "PARAMETER ERROR: meta_batch_size must be ≥ 1")]
fn test_invalid_parameter_batch_size_zero() {
    println!("🧪 Testing parameter validation: meta_batch_size zero");
    
    let device = NdArrayDevice::Cpu;
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let config = TrainingConfig {
        meta_batch_size: 1, // Config value doesn't matter, we pass 0 to sample
        tasks_per_batch: 1,
        max_samples_per_task: 100,
        min_samples_per_task: 10,
        num_features_range: (3, 3),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 50,
        early_stopping_patience: 10,
        use_gradient_checkpointing: false,
        layer_dropout_min_layers: Some(2),
        cache_trainset_representations: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let mut rng = StdRng::seed_from_u64(999);
    
    // This should panic due to meta_batch_size = 0
    prior.sample(10, 0, &device, &rng_context, &mut rng);
}
//! Dataset prior shape verification tests
//! 
//! These tests verify that all dataset providers return tensors with the correct
//! canonical shapes and dtypes as specified in the TabPFN requirements:
//! - features: [S,B,F] f32 
//! - targets: [S,B] i64
//! - train_mask: [S,B] bool
//! - labels_for_model: [S,B] i64 with -1 at test positions

use burn::{
    prelude::Backend,
};
use burn_ndarray::NdArray;

// Use the same test backend as other tests  
type TestBackend = NdArray<f32>;

use tab_pfn_rs::tabpfn::architectures::base::{
    train::{TrainingConfig, DatasetPrior, PriorType, SyntheticTabularDataset},
    transformer::DeterministicRngContext,
    config::ModelConfig,
};
use rand::{rngs::StdRng, SeedableRng};

/// Test that Gaussian prior returns canonical shapes - EXPECTED TO FAIL INITIALLY
#[test]
fn test_gaussian_prior_canonical_shapes() {
    println!("🔴 Test Gaussian prior canonical shapes - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (5, 8),
        num_classes_range: (2, 4), 
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 15,
        max_samples_per_task: 15,
        gradient_accumulation_steps: 1,
        validation_frequency: 1, 
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 3,
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(
        15, // S = num_samples 
        3,  // B = meta_batch_size
        &device,
        &rng_context,
        &mut rng
    );
    
    // Expected canonical dimensions
    let S = 15; // num_samples
    let B = 3;  // meta_batch_size
    let F = dataset.features.dims()[2]; // num_features (varies within range)
    
    // CRITICAL ASSERTIONS: These will fail if datasets don't use canonical [S,B,*] layout
    assert_eq!(dataset.features.dims(), vec![S, B, F], 
               "Gaussian prior features must be [S={},B={},F={}], got {:?}", S, B, F, dataset.features.dims());
    assert_eq!(dataset.targets.dims(), vec![S, B],
               "Gaussian prior targets must be [S={},B={}], got {:?}", S, B, dataset.targets.dims());
    assert_eq!(dataset.train_mask.dims(), vec![S, B],
               "Gaussian prior train_mask must be [S={},B={}], got {:?}", S, B, dataset.train_mask.dims());
    assert_eq!(dataset.labels_for_model.dims(), vec![S, B],
               "Gaussian prior labels_for_model must be [S={},B={}], got {:?}", S, B, dataset.labels_for_model.dims());
    
    // Verify feature range constraint
    assert!(F >= 5 && F <= 8, "num_features {} must be in range [5,8]", F);
    
    println!("✅ Gaussian prior returned canonical shapes: features[{},{},{}], targets[{},{}]", S, B, F, S, B);
}

/// Test that BayesianNN prior returns canonical shapes - EXPECTED TO FAIL INITIALLY  
#[test]
fn test_bayesian_nn_prior_canonical_shapes() {
    println!("🔴 Test BayesianNN prior canonical shapes - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(123, device.clone());
    let mut rng = StdRng::seed_from_u64(123);
    
    let config = TrainingConfig {
        prior_type: PriorType::BayesianNN,
        num_features_range: (3, 6),
        num_classes_range: (2, 5),
        feature_noise_level: 0.05,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0, 
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 20,
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
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(
        20, // S = num_samples
        2,  // B = meta_batch_size  
        &device,
        &rng_context,
        &mut rng
    );
    
    // Expected canonical dimensions
    let S = 20; // num_samples 
    let B = 2;  // meta_batch_size
    let F = dataset.features.dims()[2]; // num_features (varies within range)
    
    // CRITICAL ASSERTIONS: These will fail if datasets don't use canonical [S,B,*] layout  
    assert_eq!(dataset.features.dims(), vec![S, B, F],
               "BayesianNN prior features must be [S={},B={},F={}], got {:?}", S, B, F, dataset.features.dims());
    assert_eq!(dataset.targets.dims(), vec![S, B],
               "BayesianNN prior targets must be [S={},B={}], got {:?}", S, B, dataset.targets.dims());
    assert_eq!(dataset.train_mask.dims(), vec![S, B],
               "BayesianNN prior train_mask must be [S={},B={}], got {:?}", S, B, dataset.train_mask.dims());
    assert_eq!(dataset.labels_for_model.dims(), vec![S, B],
               "BayesianNN prior labels_for_model must be [S={},B={}], got {:?}", S, B, dataset.labels_for_model.dims());
    
    // Verify feature range constraint
    assert!(F >= 3 && F <= 6, "num_features {} must be in range [3,6]", F);
    
    println!("✅ BayesianNN prior returned canonical shapes: features[{},{},{}], targets[{},{}]", S, B, F, S, B);
}

/// Test that RandomForest prior returns canonical shapes - EXPECTED TO FAIL INITIALLY
#[test]  
fn test_random_forest_prior_canonical_shapes() {
    println!("🔴 Test RandomForest prior canonical shapes - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(456, device.clone());
    let mut rng = StdRng::seed_from_u64(456);
    
    let config = TrainingConfig {
        prior_type: PriorType::RandomForest,
        num_features_range: (4, 7),
        num_classes_range: (3, 6), 
        feature_noise_level: 0.2,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1, 
        tasks_per_batch: 1,
        min_samples_per_task: 12,
        max_samples_per_task: 12,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 4,
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(
        12, // S = num_samples
        4,  // B = meta_batch_size 
        &device,
        &rng_context,
        &mut rng
    );
    
    // Expected canonical dimensions
    let S = 12; // num_samples
    let B = 4;  // meta_batch_size
    let F = dataset.features.dims()[2]; // num_features (varies within range)
    
    // CRITICAL ASSERTIONS: These will fail if datasets don't use canonical [S,B,*] layout
    assert_eq!(dataset.features.dims(), vec![S, B, F],
               "RandomForest prior features must be [S={},B={},F={}], got {:?}", S, B, F, dataset.features.dims());
    assert_eq!(dataset.targets.dims(), vec![S, B],
               "RandomForest prior targets must be [S={},B={}], got {:?}", S, B, dataset.targets.dims());
    assert_eq!(dataset.train_mask.dims(), vec![S, B],
               "RandomForest prior train_mask must be [S={},B={}], got {:?}", S, B, dataset.train_mask.dims());
    assert_eq!(dataset.labels_for_model.dims(), vec![S, B],
               "RandomForest prior labels_for_model must be [S={},B={}], got {:?}", S, B, dataset.labels_for_model.dims());
    
    // Verify feature range constraint
    assert!(F >= 4 && F <= 7, "num_features {} must be in range [4,7]", F);
    
    println!("✅ RandomForest prior returned canonical shapes: features[{},{},{}], targets[{},{}]", S, B, F, S, B);
}

/// Test labels_for_model construction with -1 at test positions - EXPECTED TO FAIL INITIALLY  
#[test]
fn test_labels_for_model_construction() {
    println!("🔴 Test labels_for_model construction rule - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(789, device.clone());
    let mut rng = StdRng::seed_from_u64(789);
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3), // Fixed for testing
        num_classes_range: (3, 3), // Fixed for testing  
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001,
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
        meta_batch_size: 1, // Single task for easier testing
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(
        10, // S = num_samples 
        1,  // B = meta_batch_size
        &device,
        &rng_context,
        &mut rng
    );
    
    // Verify canonical shapes first
    assert_eq!(dataset.features.dims(), vec![10, 1, 3]);
    assert_eq!(dataset.targets.dims(), vec![10, 1]);
    assert_eq!(dataset.train_mask.dims(), vec![10, 1]);
    assert_eq!(dataset.labels_for_model.dims(), vec![10, 1]);
    
    // CRITICAL TEST: Verify labels_for_model construction rule
    // labels_for_model = targets.mask_where(!train_mask, -1)
    let targets_data = dataset.targets.clone().to_data();
    let mask_data = dataset.train_mask.clone().to_data(); 
    let labels_data = dataset.labels_for_model.clone().to_data();
    
    let targets_slice = targets_data.as_slice::<i64>().expect("targets should be i64");
    let mask_slice = mask_data.as_slice::<bool>().expect("train_mask should be bool");
    let labels_slice = labels_data.as_slice::<i64>().expect("labels_for_model should be i64");
    
    for i in 0..10 {
        if mask_slice[i] { 
            // Training position: labels_for_model should equal targets
            assert_eq!(labels_slice[i], targets_slice[i],
                       "At train position {}: labels_for_model[{}] = {} should equal targets[{}] = {}", 
                       i, i, labels_slice[i], i, targets_slice[i]);
            // Training labels should be valid class indices (0 to num_classes-1)
            assert!(labels_slice[i] >= 0 && labels_slice[i] < 3,
                    "Training label at position {} should be in [0,2], got {}", i, labels_slice[i]);
        } else {
            // Test position: labels_for_model should be -1
            assert_eq!(labels_slice[i], -1,
                       "At test position {}: labels_for_model[{}] = {} should be -1", 
                       i, i, labels_slice[i]);
        }
    }
    
    println!("✅ labels_for_model construction rule verified");
}

/// Test that all priors have mixed train/test positions - EXPECTED TO FAIL INITIALLY
#[test] 
fn test_train_test_split_requirements() {
    println!("🔴 Test train/test split requirements - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(999, device.clone());
    let mut rng = StdRng::seed_from_u64(999);
    
    let base_config = TrainingConfig {
        prior_type: PriorType::Gaussian, // Will be overridden
        num_features_range: (4, 4), // Fixed for testing
        num_classes_range: (2, 2), // Fixed for testing
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        tasks_per_batch: 1,
        min_samples_per_task: 20,
        max_samples_per_task: 20, 
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        meta_batch_size: 3, // Test multiple tasks
        use_gradient_checkpointing: false,
    };
    
    // Test each prior type
    for prior_type in [PriorType::Gaussian, PriorType::BayesianNN, PriorType::RandomForest] {
        println!("Testing {:?} prior train/test split requirements", prior_type);
        
        let mut config = base_config.clone();
        config.prior_type = prior_type;
        
        let prior = DatasetPrior::new(&config);
        let dataset = prior.sample::<TestBackend>(
            20, // S = num_samples
            3,  // B = meta_batch_size
            &device, 
            &rng_context,
            &mut rng
        );
        
        // Verify canonical shapes
        assert_eq!(dataset.train_mask.dims(), vec![20, 3]);
        
        // CRITICAL TEST: Each task (batch element) must have both train and test examples
        let mask_data = dataset.train_mask.clone().to_data();
        let mask_slice = mask_data.as_slice::<bool>().expect("train_mask should be bool");
        
        for task_idx in 0..3 { // For each task in meta-batch
            let mut has_train = false;
            let mut has_test = false;
            
            for sample_idx in 0..20 { // For each sample in sequence
                let linear_idx = sample_idx * 3 + task_idx; // [S,B] linear indexing
                if mask_slice[linear_idx] {
                    has_train = true;
                } else {
                    has_test = true;
                }
            }
            
            assert!(has_train, 
                    "{:?} prior task {} must have at least one training example", prior_type, task_idx);
            assert!(has_test,
                    "{:?} prior task {} must have at least one test example", prior_type, task_idx);
        }
        
        println!("✅ {:?} prior satisfies train/test split requirements", prior_type);
    }
}

/// Test dtype consistency across all tensors - EXPECTED TO FAIL INITIALLY
#[test]
fn test_tensor_dtype_consistency() {
    println!("🔴 Test tensor dtype consistency - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(555, device.clone());
    let mut rng = StdRng::seed_from_u64(555);
    
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
    
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample::<TestBackend>(
        8,  // S = num_samples
        2,  // B = meta_batch_size 
        &device,
        &rng_context,
        &mut rng
    );
    
    // Verify canonical shapes
    assert_eq!(dataset.features.dims(), vec![8, 2, 5]);
    assert_eq!(dataset.targets.dims(), vec![8, 2]);
    assert_eq!(dataset.train_mask.dims(), vec![8, 2]);
    assert_eq!(dataset.labels_for_model.dims(), vec![8, 2]);
    
    // CRITICAL TEST: Verify dtype requirements
    // Note: In Burn, we can't directly check dtypes at compile time easily,
    // but we can verify behavioral properties
    
    // Features should be f32 (test by converting to data)  
    let features_data = dataset.features.clone().to_data();
    let _features_f32 = features_data.as_slice::<f32>().expect("features should be f32");
    
    // Targets should be i64 integer type (test by converting to data)
    let targets_data = dataset.targets.clone().to_data(); 
    let targets_i64 = targets_data.as_slice::<i64>().expect("targets should be i64");
    
    // Verify target values are valid class indices [0, num_classes)
    for &target_val in targets_i64 {
        assert!(target_val >= 0 && target_val < 3,
                "Target value {} must be in range [0, 2] for 3-class problem", target_val);
    }
    
    // Train mask should be bool
    let mask_data = dataset.train_mask.clone().to_data();
    let _mask_bool = mask_data.as_slice::<bool>().expect("train_mask should be bool");
    
    // Labels for model should be i64 integer type  
    let labels_data = dataset.labels_for_model.clone().to_data();
    let labels_i64 = labels_data.as_slice::<i64>().expect("labels_for_model should be i64");
    
    // Verify labels_for_model values: either valid class indices or -1
    for &label_val in labels_i64 {
        assert!(label_val == -1 || (label_val >= 0 && label_val < 3),
                "labels_for_model value {} must be -1 or in range [0, 2]", label_val);
    }
    
    println!("✅ Tensor dtype consistency verified");
}
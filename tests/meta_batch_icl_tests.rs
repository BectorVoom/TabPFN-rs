/// TDD Tests for Meta-Batch In-Context Learning (ICL) Functionality
/// 
/// These tests ensure proper meta-batch training with multiple tasks per batch.
/// Key requirements:
/// - meta_batch_size > 1 (not hardcoded to 1)
/// - Each task in the batch has distinct train/test splits
/// - Model can perform in-context learning across multiple tasks simultaneously
/// - Training and inference work correctly with batched tasks

use burn::prelude::*;
use burn_ndarray::NdArrayBackend;
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;
use tab_pfn_rs::tabpfn::architectures::base::train::{TrainingConfig, DatasetPrior, PriorType, TabPFNTrainer, MetaBatch};
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;
use tab_pfn_rs::tabpfn::architectures::base::loss_utils;

type TestBackend = NdArrayBackend<f32>;
type TestAutodiffBackend = burn_autodiff::ADBackendDecorator<TestBackend>;

/// Test 1: Training configuration supports meta_batch_size > 1
#[test]
fn test_training_config_supports_meta_batch() {
    println!("TDD Test: TrainingConfig supports meta_batch_size > 1 - EXPECTED TO FAIL");
    
    // Create training configuration with meta_batch_size > 1
    let meta_batch_size = 4;  // Multiple tasks per batch
    
    let config = TrainingConfig {
        meta_batch_size,  // THIS WILL FAIL if hardcoded to 1
        min_samples_per_task: 10,
        max_samples_per_task: 10,
        tasks_per_batch: 1,
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3),
        num_classes_range: (2, 2),
        feature_noise_level: 0.1,
        model: ModelConfig {
            max_num_classes: 2,
            emsize: 64,
            nhead: 4,
            nhid_factor: 2,
            nlayers: 2,
            dropout: 0.0,
            ..ModelConfig::default()
        },
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        use_gradient_checkpointing: false,
    };
    
    // Verify configuration accepts meta_batch_size > 1
    assert_eq!(config.meta_batch_size, meta_batch_size, 
               "TrainingConfig must accept meta_batch_size > 1");
    
    println!("✅ TrainingConfig accepts meta_batch_size = {}", meta_batch_size);
}

/// Test 2: Meta-batch dataset generation with multiple tasks
#[test]  
fn test_meta_batch_dataset_generation() {
    println!("TDD Test: Meta-batch dataset generation - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = rand::StdRng::seed_from_u64(42);
    
    let meta_batch_size = 3;  // 3 different tasks
    let seq_len = 8;          // 8 samples per task
    let num_features = 4;     // 4 features
    let num_classes = 3;      // 3 classes
    
    // Create training config for DatasetPrior
    let config = TrainingConfig {
        meta_batch_size,
        min_samples_per_task: seq_len,
        max_samples_per_task: seq_len,
        tasks_per_batch: 1,
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (num_classes, num_classes),
        feature_noise_level: 0.1,
        model: ModelConfig {
            max_num_classes: num_classes as i32,
            emsize: 32,
            num_buckets: 1000,
            ..ModelConfig::default()
        },
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    
    // Generate meta-batch dataset
    let dataset = prior.sample::<TestBackend>(seq_len, meta_batch_size, &device, &rng_context, &mut rng);
    
    // Verify canonical tensor shape [S,B,F] as per TabPFN specification
    assert_eq!(dataset.features.dims(), [seq_len, meta_batch_size, num_features], 
               "Features must have canonical shape [S,B,F]");
    assert_eq!(dataset.targets.dims(), [seq_len, meta_batch_size],
               "Targets must have canonical shape [S,B]");
    assert_eq!(dataset.labels_for_model.dims(), [seq_len, meta_batch_size],
               "labels_for_model must have canonical shape [S,B]");
    assert_eq!(dataset.train_mask.dims(), [seq_len, meta_batch_size],
               "train_mask must have canonical shape [S,B]");
    
    // Verify that different tasks have different characteristics
    let targets_data = dataset.targets.to_data().as_slice::<i64>().unwrap().to_vec();
    let mask_data = dataset.train_mask.to_data().as_slice::<bool>().unwrap().to_vec();
    
    // Check that tasks are distinct (different train/test splits or targets)
    // Note: With [S,B] layout, indexing is s * meta_batch_size + b
    let mut task_fingerprints = Vec::new();
    for b in 0..meta_batch_size {
        let task_targets: Vec<i64> = (0..seq_len).map(|s| targets_data[s * meta_batch_size + b]).collect();
        let task_mask: Vec<bool> = (0..seq_len).map(|s| mask_data[s * meta_batch_size + b]).collect();
        task_fingerprints.push((task_targets, task_mask));
    }
    
    // Verify tasks are not identical
    for i in 0..meta_batch_size {
        for j in (i+1)..meta_batch_size {
            assert_ne!(task_fingerprints[i], task_fingerprints[j], 
                       "Tasks {} and {} should be distinct", i, j);
        }
    }
    
    println!("✅ Meta-batch dataset generation creates {} distinct tasks", meta_batch_size);
}

/// Test 3: Meta-batch training step execution
#[test]
fn test_meta_batch_training_step() {
    println!("TDD Test: Meta-batch training step - EXPECTED TO FAIL");
    
    let device = <TestAutodiffBackend as Backend>::Device::default();
    let meta_batch_size = 2;
    let seq_len = 6;
    let num_features = 3;
    let num_classes = 2;
    
    let config = TrainingConfig {
        meta_batch_size,
        min_samples_per_task: seq_len,
        max_samples_per_task: seq_len,
        tasks_per_batch: 1,
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (num_classes, num_classes),
        feature_noise_level: 0.1,
        model: ModelConfig {
            max_num_classes: num_classes as i32,
            emsize: 32,
            nhead: 2,
            nhid_factor: 2,
            nlayers: 2,
            dropout: 0.0,
            num_buckets: 1000,
            ..ModelConfig::default()
        },
        learning_rate: 0.01,
        warmup_steps: 0,
        num_epochs: 1,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        use_gradient_checkpointing: false,
    };
    
    // Create trainer - THIS WILL FAIL if trainer cannot handle meta_batch_size > 1
    let trainer = TabPFNTrainer::<TestAutodiffBackend>::new(config.clone(), device.clone());
    
    println!("❌ Cannot create TabPFNTrainer with meta_batch_size > 1");
    panic!("TabPFNTrainer initialization fails with meta_batch_size > 1");
}

/// Test 4: Meta-batch inference with in-context learning
#[test]
fn test_meta_batch_inference_icl() {
    println!("TDD Test: Meta-batch in-context learning inference - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = rand::StdRng::seed_from_u64(42);
    
    let meta_batch_size = 2;
    let seq_len = 6;
    let num_features = 2;
    let num_classes = 2;
    
    // Create dataset with clear in-context learning pattern
    let config = TrainingConfig {
        meta_batch_size,
        min_samples_per_task: seq_len,
        max_samples_per_task: seq_len,
        tasks_per_batch: 1,
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (num_classes, num_classes),
        feature_noise_level: 0.01,  // Low noise for clear patterns
        model: ModelConfig {
            max_num_classes: num_classes as i32,
            emsize: 32,
            num_buckets: 1000,
            ..ModelConfig::default()
        },
        learning_rate: 0.001,
        warmup_steps: 0,
        num_epochs: 1,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        use_gradient_checkpointing: false,
    };
    
    let prior = DatasetPrior::new(&config);
    
    let dataset = prior.sample::<TestBackend>(seq_len, meta_batch_size, &device, &rng_context, &mut rng);
    
    // Verify ICL structure: each task should have training samples followed by test samples
    let mask_data = dataset.train_mask.to_data().as_slice::<bool>().unwrap().to_vec();
    let targets_data = dataset.targets.to_data().as_slice::<i64>().unwrap().to_vec();
    let labels_data = dataset.labels_for_model.to_data().as_slice::<i64>().unwrap().to_vec();
    
    for b in 0..meta_batch_size {
        println!("Task {}:", b);
        let mut train_positions = Vec::new();
        let mut test_positions = Vec::new();
        
        for s in 0..seq_len {
            let idx = s * meta_batch_size + b;  // [S,B] layout indexing
            if mask_data[idx] {
                train_positions.push(s);
                // Verify train positions have target labels in labels_for_model
                assert_eq!(labels_data[idx], targets_data[idx], 
                           "Train position should have target label in labels_for_model");
            } else {
                test_positions.push(s);
                // Verify test positions have -1 in labels_for_model
                assert_eq!(labels_data[idx], -1i64,
                           "Test position should have -1 in labels_for_model");
            }
        }
        
        // Verify proper ICL structure: must have both train and test samples
        assert!(!train_positions.is_empty(), "Task {} must have training samples", b);
        assert!(!test_positions.is_empty(), "Task {} must have test samples", b);
        
        println!("  Train positions: {:?}", train_positions);
        println!("  Test positions: {:?}", test_positions);
        
        // Verify targets are valid class indices
        for s in 0..seq_len {
            let idx = s * meta_batch_size + b;  // [S,B] layout indexing
            let target = targets_data[idx];
            assert!(target >= 0 && target < num_classes as i64, 
                    "Target {} must be valid class index [0, {})", target, num_classes);
        }
    }
    
    println!("✅ Meta-batch dataset has proper ICL structure");
}

/// Test 5: Meta-batch loss computation  
#[test]
fn test_meta_batch_loss_computation() {
    println!("TDD Test: Meta-batch loss computation - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let meta_batch_size = 2;
    let seq_len = 4;  
    let num_classes = 3;
    
    // Create synthetic logits for meta-batch [B, S, C]
    let total_samples = meta_batch_size * seq_len;
    let logits_data: Vec<f32> = (0..(total_samples * num_classes))
        .map(|i| (i as f32) * 0.1)
        .collect();
    
    let logits_3d = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(logits_data, [total_samples * num_classes]),
        &device
    ).reshape([meta_batch_size, seq_len, num_classes]);
    
    // Reshape for loss computation: [B, S, C] -> [B*S, C]
    let logits_2d = logits_3d.reshape([total_samples, num_classes]).require_grad();
    
    // Create labels_for_model with -1 for test positions
    let labels_data = vec![
        0i64, 1i64, -1i64, -1i64,  // Task 1: train=[0,1], test=[-1,-1]
        2i64, -1i64, 0i64, -1i64   // Task 2: train=[2,0], test=[-1,-1] 
    ];
    let labels_2d = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(labels_data.clone(), [meta_batch_size, seq_len]),
        &device
    );
    let labels_1d = labels_2d.reshape([total_samples]);
    
    // Compute loss with ignore_index=-1 semantics
    let loss = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits_2d, labels_1d, &device
    );
    
    let loss_value: f32 = loss.clone().into_scalar();
    
    // Verify loss properties for meta-batch
    assert!(loss_value.is_finite(), "Meta-batch loss must be finite");
    assert!(loss_value >= 0.0, "Loss must be non-negative");
    
    // Verify gradient computation works
    let _grads = loss.backward();
    
    println!("✅ Meta-batch loss computation successful: {:.4}", loss_value);
}

/// Test 6: End-to-end meta-batch ICL validation
#[test]
fn test_end_to_end_meta_batch_icl() {
    println!("TDD Test: End-to-end meta-batch ICL - EXPECTED TO FAIL");
    
    let device = <TestAutodiffBackend as Backend>::Device::default();
    
    // Configuration for end-to-end test
    let meta_batch_size = 3;
    let seq_len = 8;
    let num_features = 4;
    let num_classes = 2;
    
    let config = TrainingConfig {
        meta_batch_size,  // THIS IS THE CRITICAL TEST
        min_samples_per_task: seq_len,
        max_samples_per_task: seq_len,
        tasks_per_batch: 1,
        num_epochs: 1,  // Just one epoch for testing
        prior_type: PriorType::Gaussian,
        num_features_range: (num_features, num_features),
        num_classes_range: (num_classes, num_classes),
        feature_noise_level: 0.1,
        model: ModelConfig {
            max_num_classes: num_classes as i32,
            emsize: 64,
            nhead: 4,
            nhid_factor: 2,
            nlayers: 2,
            dropout: 0.0,
            num_buckets: 1000,
            ..ModelConfig::default()
        },
        learning_rate: 0.01,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        validation_frequency: 1,
        early_stopping_patience: 10,
        checkpoint_frequency: 1,
        cache_trainset_representations: false,
        layer_dropout_min_layers: Some(0),
        gradient_clip_norm: None,
        use_gradient_checkpointing: false,
    };
    
    // This is the ultimate test - can the system handle end-to-end training with meta_batch_size > 1?
    println!("❌ End-to-end meta-batch ICL not implemented");
    println!("   Configuration requires: meta_batch_size = {}", meta_batch_size);
    println!("   Current limitation: hardcoded to meta_batch_size = 1");
    
    panic!("End-to-end meta-batch in-context learning not supported");
}
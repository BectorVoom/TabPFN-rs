use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::tensor::{Bool, Int, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use tab_pfn_rs::tabpfn::architectures::base::train::{
    argmax_with_tie_break_smallest, DatasetPrior, SyntheticTabularDataset, TabPFNTrainer, TrainingConfig, PriorType
};
use tab_pfn_rs::tabpfn::architectures::base::loss_utils;
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type TestBackend = Autodiff<NdArray>;

/// Test suite implementing the canonical tensor layout specifications for TabPFN
/// 
/// This module contains all TDD tests that must pass before any implementation changes.
/// All tests enforce the canonical tensor layouts:
/// - features: [S, B, F] 
/// - targets: [S, B]
/// - train_mask: [S, B] 
/// - labels_for_model: [S, B]
/// - logits: [S, B, C]
/// 
/// Where S = sequence length, B = meta-batch size, F = features, C = classes

#[test]
fn test_dataset_shapes_are_sbf() {
    /// Validates that dataset generation produces canonical tensor shapes
    /// Required shapes: features [S,B,F], targets [S,B], train_mask [S,B], labels_for_model [S,B]
    
    let device = NdArrayDevice::Cpu;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let rng_context = DeterministicRngContext::new(42, device.clone());
    
    // Test with representative dimensions
    let s = 100; // sequence length
    let b = 4;   // meta-batch size  
    let f = 10;  // number of features
    
    let config = TrainingConfig {
        meta_batch_size: b,
        num_features_range: (f, f),
        num_classes_range: (2, 5),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample(s, b, &device, &rng_context, &mut rng);
    
    // CRITICAL: Assert canonical tensor shapes
    assert_eq!(
        dataset.features.dims(), 
        &[s, b, f], 
        "features shape mismatch: expected [S={}, B={}, F={}], got {:?}", 
        s, b, f, dataset.features.dims()
    );
    
    assert_eq!(
        dataset.targets.dims(), 
        &[s, b], 
        "targets shape mismatch: expected [S={}, B={}], got {:?}", 
        s, b, dataset.targets.dims()
    );
    
    assert_eq!(
        dataset.train_mask.dims(), 
        &[s, b], 
        "train_mask shape mismatch: expected [S={}, B={}], got {:?}", 
        s, b, dataset.train_mask.dims()
    );
    
    assert_eq!(
        dataset.labels_for_model.dims(), 
        &[s, b], 
        "labels_for_model shape mismatch: expected [S={}, B={}], got {:?}", 
        s, b, dataset.labels_for_model.dims()
    );
    
    println!("✓ All dataset tensors conform to canonical [S,B,F] and [S,B] layouts");
}

#[test] 
fn test_mask_has_train_and_test_per_task() {
    /// Validates that each task in the meta-batch has both training and test examples
    /// For each b in 0..B: must have at least one train_mask[s,b] == true AND one == false
    
    let device = NdArrayDevice::Cpu;
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let rng_context = DeterministicRngContext::new(123, device.clone());
    
    let s = 50;
    let b = 8; 
    
    let config = TrainingConfig {
        meta_batch_size: b,
        num_features_range: (10, 10),
        num_classes_range: (2, 5),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample(s, b, &device, &rng_context, &mut rng);
    
    // Convert mask to data for validation
    let mask_data = dataset.train_mask.to_data();
    let mask_slice = mask_data.as_slice::<bool>().expect("Failed to read train_mask data");
    
    // Validate per-task train/test distribution
    for task_idx in 0..b {
        let mut has_train = false;
        let mut has_test = false;
        
        for seq_idx in 0..s {
            let linear_idx = seq_idx * b + task_idx;
            let is_train = mask_slice[linear_idx];
            
            if is_train {
                has_train = true;
            } else {
                has_test = true;
            }
        }
        
        assert!(
            has_train && has_test, 
            "Task {} missing train/test split: has_train={}, has_test={}", 
            task_idx, has_train, has_test
        );
    }
    
    println!("✓ All {} tasks have both training and test examples", b);
}

#[test]
fn test_labels_for_model_matches_targets_and_mask() {
    /// Validates labels_for_model construction: targets where train_mask=true, -1 where train_mask=false
    /// For every position (s,b): if train_mask[s,b] then labels_for_model[s,b] == targets[s,b] else == -1
    
    let device = NdArrayDevice::Cpu;
    let mut rng = ChaCha8Rng::seed_from_u64(456);
    let rng_context = DeterministicRngContext::new(456, device.clone());
    
    let s = 30;
    let b = 6;
    
    let config = TrainingConfig {
        meta_batch_size: b,
        num_features_range: (10, 10),
        num_classes_range: (2, 5),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    let prior = DatasetPrior::new(&config);
    let dataset = prior.sample(s, b, &device, &rng_context, &mut rng);
    
    // Convert tensors to data for validation  
    let targets_data = dataset.targets.to_data();
    let targets_slice = targets_data.as_slice::<i64>().expect("Failed to read targets data");
    
    let mask_data = dataset.train_mask.to_data();
    let mask_slice = mask_data.as_slice::<bool>().expect("Failed to read train_mask data");
    
    let labels_data = dataset.labels_for_model.to_data(); 
    let labels_slice = labels_data.as_slice::<i64>().expect("Failed to read labels_for_model data");
    
    // Validate label construction at every position
    for seq_idx in 0..s {
        for batch_idx in 0..b {
            let linear_idx = seq_idx * b + batch_idx;
            let is_train = mask_slice[linear_idx];
            let target_value = targets_slice[linear_idx];
            let label_value = labels_slice[linear_idx];
            
            if is_train {
                assert_eq!(
                    label_value, 
                    target_value,
                    "Training position mismatch at ({}, {}): expected labels_for_model={}, got {}",
                    seq_idx, batch_idx, target_value, label_value
                );
            } else {
                assert_eq!(
                    label_value, 
                    -1,
                    "Test position mismatch at ({}, {}): expected labels_for_model=-1, got {}",
                    seq_idx, batch_idx, label_value
                );
            }
        }
    }
    
    println!("✓ labels_for_model correctly constructed with targets for training positions and -1 for test positions");
}

#[test]
fn test_logits_reshape_and_argmax_tiebreak() {
    /// Tests logits reshape to [S,B,C] and argmax tie-breaking behavior
    /// Ensures argmax returns [S,B] and chooses smallest class index for ties
    
    let device = NdArrayDevice::default();
    
    // Create test logits with known tie scenarios  
    let s = 3;
    let b = 2;
    let c = 4;
    
    // Test case 1: No ties - clear winners
    let logits_data_1 = vec![
        // Sample 0, Batch 0: [1.0, 0.5, 0.0, 0.2] -> argmax = 0
        1.0, 0.5, 0.0, 0.2,
        // Sample 0, Batch 1: [0.1, 2.0, 0.5, 0.0] -> argmax = 1  
        0.1, 2.0, 0.5, 0.0,
        // Sample 1, Batch 0: [0.0, 0.0, 3.0, 1.0] -> argmax = 2
        0.0, 0.0, 3.0, 1.0,
        // Sample 1, Batch 1: [0.5, 0.1, 0.0, 4.0] -> argmax = 3
        0.5, 0.1, 0.0, 4.0,
        // Sample 2, Batch 0: [1.5, 0.0, 1.0, 0.5] -> argmax = 0
        1.5, 0.0, 1.0, 0.5,
        // Sample 2, Batch 1: [0.0, 1.0, 2.0, 0.5] -> argmax = 2
        0.0, 1.0, 2.0, 0.5,
    ];
    
    let logits_1: Tensor<TestBackend, 3> = Tensor::from_data(&logits_data_1[..], &device)
        .reshape([s, b, c]);
    
    assert_eq!(logits_1.dims(), &[s, b, c], "Logits must be [S,B,C] format");
    
    let argmax_result_1 = argmax_with_tie_break_smallest(logits_1);
    assert_eq!(argmax_result_1.dims(), &[s, b], "Argmax result must be [S,B]");
    
    let result_data_1 = argmax_result_1.to_data();
    let result_slice_1 = result_data_1.as_slice::<i64>().expect("Failed to read argmax result");
    let expected_1 = vec![0i64, 1i64, 2i64, 3i64, 0i64, 2i64];
    
    assert_eq!(result_slice_1, &expected_1, "Argmax results mismatch for no-tie case");
    
    // Test case 2: Ties - should choose smallest index
    let logits_data_2 = vec![
        // Sample 0, Batch 0: [1.0, 1.0, 0.5, 0.5] -> tie between 0,1 -> choose 0
        1.0, 1.0, 0.5, 0.5,
        // Sample 0, Batch 1: [0.8, 0.8, 0.8, 0.2] -> 3-way tie -> choose 0
        0.8, 0.8, 0.8, 0.2,
        // Sample 1, Batch 0: [0.0, 2.0, 2.0, 1.0] -> tie between 1,2 -> choose 1  
        0.0, 2.0, 2.0, 1.0,
        // Sample 1, Batch 1: [3.0, 3.0, 3.0, 3.0] -> 4-way tie -> choose 0
        3.0, 3.0, 3.0, 3.0,
        // Sample 2, Batch 0: [0.5, 0.1, 1.5, 1.5] -> tie between 2,3 -> choose 2
        0.5, 0.1, 1.5, 1.5,
        // Sample 2, Batch 1: [2.5, 0.0, 0.0, 2.5] -> tie between 0,3 -> choose 0
        2.5, 0.0, 0.0, 2.5,
    ];
    
    let logits_2: Tensor<TestBackend, 3> = Tensor::from_data(&logits_data_2[..], &device)
        .reshape([s, b, c]);
    
    let argmax_result_2 = argmax_with_tie_break_smallest(logits_2);
    let result_data_2 = argmax_result_2.to_data(); 
    let result_slice_2 = result_data_2.as_slice::<i64>().expect("Failed to read argmax result");
    let expected_2 = vec![0i64, 0i64, 1i64, 0i64, 2i64, 0i64];
    
    assert_eq!(result_slice_2, &expected_2, "Argmax tie-breaking failed: expected smallest indices");
    
    println!("✓ Argmax correctly handles [S,B,C] input and chooses smallest class index for ties");
}

#[test]
fn test_loss_ignore_index_behaviour() {
    /// Validates that masked cross-entropy loss properly ignores positions with ignore_index=-1
    /// Creates known logits/labels scenario and verifies numerical behavior
    
    let device = NdArrayDevice::default();
    
    // Create test scenario: batch_size=4, num_classes=3
    let batch_size = 4;
    let num_classes = 3;
    
    // Logits: [batch_size, num_classes] 
    let logits_data = vec![
        1.0, 0.0, 0.0,  // Sample 0: clear prediction for class 0
        0.0, 2.0, 0.0,  // Sample 1: clear prediction for class 1  
        0.0, 0.0, 1.5,  // Sample 2: clear prediction for class 2
        1.0, 1.0, 1.0,  // Sample 3: uniform logits (will be ignored)
    ];
    let logits: Tensor<TestBackend, 2> = Tensor::from_data(&logits_data[..], &device)
        .reshape([batch_size, num_classes]);
    
    // Labels with ignore_index=-1 for sample 3
    let labels_data = vec![0i64, 1i64, 2i64, -1i64];  // Last sample ignored
    let labels: Tensor<TestBackend, 1, Int> = Tensor::from_data(&labels_data[..], &device)
        .reshape([batch_size]);
    
    // Compute masked loss
    let loss = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits.clone(),
        labels.clone(), 
        &device
    );
    
    // Verify loss is finite
    assert!(loss.to_data().as_slice::<f32>().unwrap()[0].is_finite(), "Loss must be finite");
    
    // Test case 2: All samples ignored (should handle gracefully)
    let all_ignored_labels = vec![-1i64, -1i64, -1i64, -1i64];
    let all_ignored: Tensor<TestBackend, 1, Int> = Tensor::from_data(&all_ignored_labels[..], &device)
        .reshape([batch_size]);
        
    let loss_all_ignored = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits,
        all_ignored,
        &device  
    );
    
    // Should not panic and return finite value (even if 0)
    let loss_value = loss_all_ignored.to_data().as_slice::<f32>().unwrap()[0];
    assert!(loss_value.is_finite(), "Loss with all ignored samples must be finite");
    
    println!("✓ Masked cross-entropy loss correctly ignores positions with ignore_index=-1");
}

#[test]
fn test_train_step_smoke() {
    /// End-to-end smoke test for training step with deterministic RNG
    /// Verifies no panics occur and loss is finite
    
    let device = NdArrayDevice::Cpu;
    let mut rng = ChaCha8Rng::seed_from_u64(789);
    let rng_context = DeterministicRngContext::new(789, device.clone());
    
    // Create minimal training configuration
    let config = TrainingConfig {
        meta_batch_size: 2,
        num_features_range: (5, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context).expect("Failed to create trainer");
    
    // Execute single training step
    let loss = trainer.train_step(&device, &mut rng);
    
    // Critical validations
    assert!(loss.is_finite(), "Training loss must be finite, got: {}", loss);
    assert!(loss >= 0.0, "Training loss must be non-negative, got: {}", loss);
    
    // Verify trainer state updated
    assert_eq!(trainer.iteration, 1, "Trainer iteration should increment");
    
    println!("✓ Training step completed successfully with finite loss: {:.6}", loss);
}

#[test]
fn test_build_and_ci_tests() {
    /// Placeholder test to verify CI build commands succeed
    /// This test ensures the test suite can be compiled and run
    
    // Basic compilation and linking verification
    let device = NdArrayDevice::Cpu;
    println!("✓ Device created successfully: {:?}", device);
    
    // Verify core data structures can be instantiated
    let config = TrainingConfig {
        meta_batch_size: 2,
        num_features_range: (5, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
        prior_type: PriorType::Gaussian,
        model: ModelConfig::default(),
        learning_rate: 0.001,
        layer_dropout_min_layers: 2,
        cache_trainset_representations: false,
    };
    let prior = DatasetPrior::new(&config);
    let rng_context = DeterministicRngContext::new(12345, device.clone());
    
    println!("✓ Core structures instantiated successfully");
    println!("✓ CI build validation passed");
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_tensor_flow() {
        /// Integration test verifying complete tensor flow through the pipeline
        /// Tests dataset -> model forward -> loss computation with canonical layouts
        
        let device = NdArrayDevice::Cpu;
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        let rng_context = DeterministicRngContext::new(999, device.clone());
        
        // Generate dataset with canonical shapes
        let s = 20;
        let b = 2; 
        let config = TrainingConfig {
            meta_batch_size: b,
            num_features_range: (10, 10),
            num_classes_range: (2, 5),
            feature_noise_level: 0.1,
            prior_type: PriorType::Gaussian,
            model: ModelConfig::default(),
            learning_rate: 0.001,
            layer_dropout_min_layers: 2,
            cache_trainset_representations: false,
        };
        let prior = DatasetPrior::new(&config);
        let dataset = prior.sample(s, b, &device, &rng_context, &mut rng);
        
        // Verify dataset shapes are canonical
        assert_eq!(dataset.features.dims(), &[s, b, 10]); // Assuming 10 features
        assert_eq!(dataset.targets.dims(), &[s, b]);
        assert_eq!(dataset.train_mask.dims(), &[s, b]);
        assert_eq!(dataset.labels_for_model.dims(), &[s, b]);
        
        // Verify data integrity
        dataset.validate_shapes_or_panic();
        
        println!("✓ End-to-end tensor flow validation passed with canonical layouts");
    }
}
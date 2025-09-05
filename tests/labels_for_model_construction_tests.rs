/// TDD Tests for Exact labels_for_model Construction Rule
/// 
/// These tests enforce the CANONICAL construction rule for labels_for_model:
/// 
/// labels_for_model[b,s] = targets[b,s] if train_mask[b,s] else -1
/// 
/// CRITICAL REQUIREMENTS:
/// - targets ([B, S]): ground-truth labels for every position (0..C-1)  
/// - labels_for_model ([B, S]): labels fed to model; train positions contain targets[b,s], test positions contain -1
/// - Construction: labels_for_model = targets.mask_where(train_mask.bool_not(), neg_ones)

use burn::prelude::*;
use burn_ndarray::NdArray;
use rand::{rngs::StdRng, SeedableRng};
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;
use tab_pfn_rs::tabpfn::architectures::base::train::{DatasetPrior, PriorType, TrainingConfig};
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type TestBackend = NdArray<f32>;

/// Test 1: Exact construction rule verification with known values
#[test]
fn test_exact_labels_for_model_construction_rule() {
    println!("TDD Test: Exact labels_for_model construction rule - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test case with known targets and train mask
    let batch_size = 2; 
    let seq_len = 4;
    
    // Known targets for verification
    let targets_data = vec![
        0i64, 1i64, 2i64, 1i64,  // Task 1: [0, 1, 2, 1]
        1i64, 0i64, 1i64, 2i64   // Task 2: [1, 0, 1, 2]  
    ];
    let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(targets_data, [batch_size * seq_len]),
        &device
    ).reshape([batch_size, seq_len]);
    
    // Known train mask
    let train_mask_data = vec![
        true, true, false, false,   // Task 1: train positions 0,1; test positions 2,3
        true, false, true, false    // Task 2: train positions 0,2; test positions 1,3
    ];
    let train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data, [batch_size * seq_len]),
        &device  
    ).reshape([batch_size, seq_len]);
    
    // CANONICAL CONSTRUCTION RULE - This is the SPECIFICATION
    let neg_ones = Tensor::<TestBackend, 2, burn::tensor::Int>::ones_like(&targets) * (-1);
    let expected_labels_for_model = targets.clone().mask_where(train_mask.clone().bool_not(), neg_ones);
    
    // Expected result should be:
    // Task 1: [0, 1, -1, -1]  (positions 2,3 are test -> -1)
    // Task 2: [1, -1, 1, -1]  (positions 1,3 are test -> -1)
    let expected_data = expected_labels_for_model.to_data().as_slice::<i64>().unwrap().to_vec();
    let expected_values = vec![
        0i64, 1i64, -1i64, -1i64,  // Task 1
        1i64, -1i64, 1i64, -1i64   // Task 2
    ];
    assert_eq!(expected_data, expected_values, "Expected construction rule verification");
    
    // NOW TEST CURRENT IMPLEMENTATION
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    // Use DatasetPrior constructor 
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3),
        num_classes_range: (3, 3),
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
    
    // Generate dataset and check if it follows construction rule
    let dataset = prior.sample::<TestBackend>(seq_len, batch_size, &device, &rng_context, &mut rng);
    
    // Extract actual labels_for_model from current implementation
    let actual_labels_data = dataset.labels_for_model.to_data().as_slice::<i64>().unwrap().to_vec();
    let actual_targets_data = dataset.targets.to_data().as_slice::<i64>().unwrap().to_vec();
    let actual_train_mask_data = dataset.train_mask.to_data().as_slice::<bool>().unwrap().to_vec();
    
    // CRITICAL VERIFICATION: Check construction rule element by element
    for i in 0..actual_labels_data.len() {
        let expected_label = if actual_train_mask_data[i] {
            actual_targets_data[i]  // Train position -> target value
        } else {
            -1i64  // Test position -> -1
        };
        
        if actual_labels_data[i] != expected_label {
            println!("❌ Construction rule violation at position {}: got {}, expected {}", 
                     i, actual_labels_data[i], expected_label);
            println!("   train_mask[{}] = {}, targets[{}] = {}", 
                     i, actual_train_mask_data[i], i, actual_targets_data[i]);
            panic!("labels_for_model construction rule not followed");
        }
    }
    
    println!("✅ labels_for_model construction follows exact rule: train→targets, test→-1");
}

/// Test 2: Construction rule with different train/test splits
#[test]
fn test_construction_rule_various_splits() {
    println!("TDD Test: Construction rule with various train/test splits - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let batch_size = 3;
    let seq_len = 6;
    
    // Test different train/test patterns
    let test_patterns = vec![
        // Pattern 1: First half train, second half test
        vec![true, true, true, false, false, false,   // Task 1
             true, true, true, false, false, false,   // Task 2  
             true, true, true, false, false, false],  // Task 3
        
        // Pattern 2: Alternating train/test
        vec![true, false, true, false, true, false,   // Task 1
             false, true, false, true, false, true,   // Task 2
             true, false, true, false, true, false],  // Task 3
             
        // Pattern 3: Random pattern
        vec![true, true, false, true, false, false,   // Task 1
             false, true, true, false, true, false,   // Task 2
             true, false, false, true, true, false],  // Task 3
    ];
    
    for (pattern_idx, mask_data) in test_patterns.iter().enumerate() {
        println!("Testing pattern {}", pattern_idx + 1);
        
        // Create synthetic targets
        let targets_data: Vec<i64> = (0..(batch_size * seq_len)).map(|i| (i % 3) as i64).collect();
        let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
            TensorData::new(targets_data.clone(), [batch_size * seq_len]),
            &device
        ).reshape([batch_size, seq_len]);
        
        let train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
            TensorData::new(mask_data.clone(), [batch_size * seq_len]),
            &device
        ).reshape([batch_size, seq_len]);
        
        // Apply canonical construction rule
        let neg_ones = Tensor::<TestBackend, 2, burn::tensor::Int>::ones_like(&targets) * (-1);
        let expected_labels = targets.clone().mask_where(train_mask.clone().bool_not(), neg_ones);
        
        // Verify construction rule manually
        let expected_data = expected_labels.to_data().as_slice::<i64>().unwrap().to_vec();
        
        for i in 0..expected_data.len() {
            let expected_value = if mask_data[i] { targets_data[i] } else { -1i64 };
            assert_eq!(expected_data[i], expected_value, 
                       "Pattern {} position {}: construction rule not satisfied", 
                       pattern_idx + 1, i);
        }
    }
    
    println!("✅ Construction rule verified for various train/test split patterns");
}

/// Test 3: Construction rule with edge cases  
#[test]
fn test_construction_rule_edge_cases() {
    println!("TDD Test: Construction rule edge cases - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Edge Case 1: All positions are train positions
    let batch_size = 1;
    let seq_len = 4;
    
    let targets_data = vec![2i64, 0i64, 1i64, 2i64];
    let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(targets_data.clone(), [seq_len]),
        &device
    ).reshape([batch_size, seq_len]);
    
    let all_train_mask_data = vec![true, true, true, true];
    let all_train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(all_train_mask_data, [seq_len]),
        &device
    ).reshape([batch_size, seq_len]);
    
    let neg_ones = Tensor::<TestBackend, 2, burn::tensor::Int>::ones_like(&targets) * (-1);
    let labels_all_train = targets.clone().mask_where(all_train_mask.clone().bool_not(), neg_ones.clone());
    
    // All positions should be targets (no -1 values)
    let result_data = labels_all_train.to_data().as_slice::<i64>().unwrap().to_vec();
    assert_eq!(result_data, targets_data, "All train positions should equal targets");
    assert!(!result_data.contains(&-1i64), "No -1 values when all positions are train");
    
    // Edge Case 2: All positions are test positions
    let all_test_mask_data = vec![false, false, false, false];
    let all_test_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(all_test_mask_data, [seq_len]),
        &device
    ).reshape([batch_size, seq_len]);
    
    let labels_all_test = targets.clone().mask_where(all_test_mask.clone().bool_not(), neg_ones.clone());
    
    // All positions should be -1
    let result_data = labels_all_test.to_data().as_slice::<i64>().unwrap().to_vec();
    let expected_all_neg_ones = vec![-1i64, -1i64, -1i64, -1i64];
    assert_eq!(result_data, expected_all_neg_ones, "All test positions should be -1");
    
    // Edge Case 3: Single sample per task
    let single_sample_targets = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(vec![1i64], [1]),
        &device
    ).reshape([1, 1]);
    
    let single_train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        TensorData::new(vec![true], [1]),
        &device
    ).reshape([1, 1]);
    
    let single_neg_ones = Tensor::<TestBackend, 2, burn::tensor::Int>::ones_like(&single_sample_targets) * (-1);
    let single_labels = single_sample_targets.clone().mask_where(single_train_mask.bool_not(), single_neg_ones);
    
    let single_result = single_labels.to_data().as_slice::<i64>().unwrap().to_vec();
    assert_eq!(single_result, vec![1i64], "Single train sample should equal target");
    
    println!("✅ Construction rule handles edge cases correctly");
}

/// Test 4: Meta-batch construction rule
#[test]
fn test_meta_batch_construction_rule() {
    println!("TDD Test: Meta-batch labels_for_model construction - EXPECTED TO FAIL");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    // Test with multiple tasks (meta-batch size > 1)
    let batch_size = 4;  // 4 tasks in meta-batch
    let seq_len = 5;     // 5 samples per task
    let num_classes = 3;
    
    // Use DatasetPrior constructor
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (2, 2),
        num_classes_range: (num_classes, num_classes),
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
    
    // THIS WILL FAIL: Current implementation likely hardcoded to batch_size=1
    let dataset = prior.sample::<TestBackend>(seq_len, batch_size, &device, &rng_context, &mut rng);
    
    // Verify shapes are correct for meta-batch
    assert_eq!(dataset.targets.dims(), [batch_size, seq_len], "Targets must be [B,S] for meta-batch");
    assert_eq!(dataset.labels_for_model.dims(), [batch_size, seq_len], "labels_for_model must be [B,S] for meta-batch");
    assert_eq!(dataset.train_mask.dims(), [batch_size, seq_len], "train_mask must be [B,S] for meta-batch");
    
    // Verify construction rule for each task in meta-batch
    let targets_data = dataset.targets.to_data().as_slice::<i64>().unwrap().to_vec();
    let labels_data = dataset.labels_for_model.to_data().as_slice::<i64>().unwrap().to_vec();
    let mask_data = dataset.train_mask.to_data().as_slice::<bool>().unwrap().to_vec();
    
    for b in 0..batch_size {
        for s in 0..seq_len {
            let idx = b * seq_len + s;
            let expected_label = if mask_data[idx] { targets_data[idx] } else { -1i64 };
            
            if labels_data[idx] != expected_label {
                println!("❌ Meta-batch construction rule violation at task {}, position {}", b, s);
                println!("   Expected: {}, Got: {}, train_mask: {}, target: {}", 
                         expected_label, labels_data[idx], mask_data[idx], targets_data[idx]);
                panic!("Meta-batch labels_for_model construction rule failed");
            }
        }
    }
    
    // Additional validation: Check that we have both train and test positions
    let num_train_positions = mask_data.iter().filter(|&&x| x).count();
    let num_test_positions = mask_data.iter().filter(|&&x| !x).count();
    
    assert!(num_train_positions > 0, "Meta-batch must have at least some train positions");
    assert!(num_test_positions > 0, "Meta-batch must have at least some test positions");
    
    let num_neg_ones = labels_data.iter().filter(|&&x| x == -1).count();
    assert_eq!(num_neg_ones, num_test_positions, "Number of -1 labels must equal number of test positions");
    
    println!("❌ Current implementation cannot handle meta-batch construction");
    panic!("Meta-batch labels_for_model construction not implemented");
}
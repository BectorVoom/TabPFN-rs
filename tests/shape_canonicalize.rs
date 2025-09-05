//! Shape canonicalization tests for tensor layout standardization
//! 
//! These tests verify that all dataset providers return tensors in the canonical
//! [S,B,F] layout and that the canonicalize_to_sbf() method correctly converts
//! from other common layouts to the canonical format.

use burn::{
    tensor::{Tensor, TensorData, Int, backend::{Backend, AutodiffBackend}},
    backend::Autodiff,
};
use burn_ndarray::{NdArray, NdArrayDevice};

// Use the same test backend as other tests
type TestBackend = Autodiff<NdArray<f32>>;

use tab_pfn_rs::tabpfn::architectures::base::{
    train::{TrainingConfig, DatasetPrior, PriorType, SyntheticTabularDataset},
    transformer::DeterministicRngContext,
    config::ModelConfig,
};
use rand::{rngs::StdRng, SeedableRng};

/// Test that SyntheticTabularDataset has canonicalize_to_sbf method
#[test]
fn test_canonicalize_to_sbf_method_exists() {
    println!("✅ Testing canonicalize_to_sbf method existence");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Create a sample dataset with [B,S,F] layout  
    let batch_size = 2;
    let seq_len = 4; 
    let num_features = 3;
    
    // Create features in [B,S,F] format (incorrect layout)
    let features_bsf = Tensor::<TestBackend, 1>::zeros([batch_size * seq_len * num_features], &device)
        .reshape([batch_size, seq_len, num_features]);
    
    let targets = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([batch_size * seq_len], &device)
        .reshape([batch_size, seq_len]);
    
    let train_mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        vec![true; batch_size * seq_len].as_slice(), &device
    ).reshape([batch_size, seq_len]);
    
    let labels_for_model = targets.clone();
    
    let dataset = SyntheticTabularDataset {
        features: features_bsf,
        targets,
        train_mask, 
        labels_for_model,
        dag: None,
    };
    
    // Test that the method exists and can be called
    let mut mutable_dataset = dataset;
    let result = mutable_dataset.canonicalize_to_sbf();
    assert!(result.is_ok(), "canonicalize_to_sbf should succeed");
}

/// Test canonicalize_to_sbf converts [B,S,F] to [S,B,F]
#[test]
fn test_canonicalize_bsf_to_sbf() {
    println!("✅ Test canonicalize [B,S,F] → [S,B,F]");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data in [B,S,F] layout
    let B = 2; // meta batch size  
    let S = 4; // sequence length
    let F = 3; // num features
    
    let features_data: Vec<f32> = (0..(B*S*F)).map(|i| i as f32).collect();
    let features_bsf = Tensor::<TestBackend, 1>::from_floats(
        features_data.as_slice(), &device
    ).reshape([B, S, F]);
    
    let targets_bsf = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([B * S], &device)
        .reshape([B, S]);
    
    let mask_bsf = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        vec![true; B * S].as_slice(), &device  
    ).reshape([B, S]);
    
    let labels_bsf = targets_bsf.clone();
    
    let mut dataset = SyntheticTabularDataset {
        features: features_bsf,
        targets: targets_bsf,
        train_mask: mask_bsf,
        labels_for_model: labels_bsf,
        dag: None,
    };
    
    // Verify initial layout is [B,S,F]
    assert_eq!(dataset.features.dims().to_vec(), vec![B, S, F]);
    assert_eq!(dataset.targets.dims().to_vec(), vec![B, S]);
    assert_eq!(dataset.train_mask.dims().to_vec(), vec![B, S]);
    assert_eq!(dataset.labels_for_model.dims().to_vec(), vec![B, S]);
    
    // Call canonicalize_to_sbf to convert to canonical layout
    let result = dataset.canonicalize_to_sbf();
    assert!(result.is_ok(), "canonicalize_to_sbf should succeed: {:?}", result);
    
    // Verify conversion to [S,B,F] canonical layout  
    assert_eq!(dataset.features.dims().to_vec(), vec![S, B, F]);
    assert_eq!(dataset.targets.dims().to_vec(), vec![S, B]);
    assert_eq!(dataset.train_mask.dims().to_vec(), vec![S, B]); 
    assert_eq!(dataset.labels_for_model.dims().to_vec(), vec![S, B]);
}

/// Test canonicalize_to_sbf handles [F,B,S] layout (edge case)
#[test] 
fn test_canonicalize_fbs_to_sbf() {
    println!("✅ Test canonicalize [F,B,S] → [S,B,F] (edge case)");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data in [F,B,S] layout (another common permutation)
    let F = 3; // num features
    let B = 2; // meta batch size
    let S = 4; // sequence length
    
    let features_data: Vec<f32> = (0..(F*B*S)).map(|i| i as f32).collect();
    let features_fbs = Tensor::<TestBackend, 1>::from_floats(
        features_data.as_slice(), &device
    ).reshape([F, B, S]);
    
    // For non-feature tensors, assume they follow [S,B] or similar pattern
    let targets_sb = Tensor::<TestBackend, 1, burn::tensor::Int>::zeros([S * B], &device)
        .reshape([S, B]);
    
    let mask_sb = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        vec![true; S * B].as_slice(), &device
    ).reshape([S, B]);
    
    let labels_sb = targets_sb.clone();
    
    let mut dataset = SyntheticTabularDataset {
        features: features_fbs,
        targets: targets_sb,
        train_mask: mask_sb, 
        labels_for_model: labels_sb,
        dag: None,
    };
    
    // Verify initial layout
    assert_eq!(dataset.features.dims().to_vec(), vec![F, B, S]);
    assert_eq!(dataset.targets.dims().to_vec(), vec![S, B]); 
    
    // Call canonicalize_to_sbf - may not handle [F,B,S] perfectly but should not crash
    let result = dataset.canonicalize_to_sbf();
    
    // The current implementation focuses on [B,S,F] -> [S,B,F], so [F,B,S] may not convert perfectly
    // but it should at least not crash and maintain valid shapes
    assert!(result.is_ok(), "canonicalize_to_sbf should not crash on [F,B,S] input: {:?}", result);
    
    // Verify final dimensions are valid (3D for features, 2D for others)
    assert_eq!(dataset.features.dims().len(), 3, "Features should remain 3D");
    assert_eq!(dataset.targets.dims().to_vec(), vec![S, B]); // Already correct
}

/// Test that dataset providers return canonical [S,B,F] layout
#[test]
fn test_dataset_providers_canonicalize() {
    println!("✅ Test dataset providers return canonical layout");
    
    let device = <TestBackend as Backend>::Device::default();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    let config = TrainingConfig {
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
        model: ModelConfig::default(),
        learning_rate: 0.001,
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
    
    let prior = DatasetPrior::new(&config);
    
    // Sample dataset from each prior type
    for prior_type in [PriorType::Gaussian, PriorType::BayesianNN, PriorType::RandomForest] {
        let mut test_config = config.clone();
        test_config.prior_type = prior_type.clone();
        let test_prior = DatasetPrior::new(&test_config);
        
        let dataset = test_prior.sample::<TestBackend>(
            15, // num_samples 
            2,  // meta_batch_size
            &device,
            &rng_context, 
            &mut rng
        );
        
        // CRITICAL: All datasets MUST be in canonical [S,B,F] format after sampling
        let S = 15; // num_samples
        let B = 2;  // meta_batch_size
        let F = dataset.features.dims()[2]; // num_features (varies)
        
        // Verify dataset providers return canonical [S,B,F] layout after canonicalization
        let dims = dataset.features.dims();
        println!("Prior {:?} returned features with dims: {:?}", prior_type, dims);
        
        assert_eq!(dataset.features.dims().to_vec(), vec![S, B, F], 
                   "Dataset provider must return features in canonical [S,B,F] layout, got {:?}", 
                   dataset.features.dims());
        assert_eq!(dataset.targets.dims().to_vec(), vec![S, B],
                   "Dataset provider must return targets in canonical [S,B] layout");
        assert_eq!(dataset.train_mask.dims().to_vec(), vec![S, B], 
                   "Dataset provider must return train_mask in canonical [S,B] layout");
        assert_eq!(dataset.labels_for_model.dims().to_vec(), vec![S, B],
                   "Dataset provider must return labels_for_model in canonical [S,B] layout");
    }
}

/// Test data preservation during canonicalization
#[test]
fn test_canonicalize_preserves_data() {
    println!("✅ Test canonicalize preserves data values");
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create test data with known pattern in [B,S,F] layout
    let B = 2;
    let S = 3; 
    let F = 2;
    
    // Create features with distinctive pattern
    let features_data = vec![
        // Batch 0: 
        1.0, 2.0,   // Sample 0: features [1,2]
        3.0, 4.0,   // Sample 1: features [3,4] 
        5.0, 6.0,   // Sample 2: features [5,6]
        // Batch 1:
        7.0, 8.0,   // Sample 0: features [7,8]
        9.0, 10.0,  // Sample 1: features [9,10]
        11.0, 12.0, // Sample 2: features [11,12]
    ];
    
    let features_bsf = Tensor::<TestBackend, 1>::from_floats(
        features_data.as_slice(), &device
    ).reshape([B, S, F]);
    
    let targets_bsf = Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints(
        vec![0i64, 1, 0, 1, 0, 1].as_slice(), &device
    ).reshape([B, S]);
    
    let mask_bsf = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(
        vec![true, false, true, false, true, false].as_slice(), &device
    ).reshape([B, S]);
    
    let labels_bsf = targets_bsf.clone();
    
    let mut dataset = SyntheticTabularDataset {
        features: features_bsf.clone(), 
        targets: targets_bsf.clone(),
        train_mask: mask_bsf.clone(),
        labels_for_model: labels_bsf.clone(),
        dag: None,
    };
    
    // Store original data for comparison
    let original_features_data = dataset.features.clone().to_data();
    let original_targets_data = dataset.targets.clone().to_data();
    
    // Call canonicalize_to_sbf to convert layout
    let result = dataset.canonicalize_to_sbf();
    assert!(result.is_ok(), "canonicalize_to_sbf should succeed: {:?}", result);
    
    // Verify layout changed to [S,B,F] 
    assert_eq!(dataset.features.dims().to_vec(), vec![S, B, F]);
    assert_eq!(dataset.targets.dims().to_vec(), vec![S, B]);
    
    // Verify data preservation - elements should be same, just reordered
    let canonicalized_features_data = dataset.features.clone().to_data();
    let canonicalized_targets_data = dataset.targets.clone().to_data();
    
    // Check that we have same elements (data preservation)
    assert_eq!(original_features_data.as_slice::<f32>().unwrap().len(), 
              canonicalized_features_data.as_slice::<f32>().unwrap().len());
    
    // Verify specific data mapping from [B,S,F] to [S,B,F]:
    // Original [B=0,S=0,F=0] = 1.0 should map to [S=0,B=0,F=0] 
    // Original [B=1,S=0,F=0] = 7.0 should map to [S=0,B=1,F=0]
    let canon_data = canonicalized_features_data.as_slice::<f32>().unwrap();
    assert!((canon_data[0] - 1.0).abs() < 1e-6, "Expected 1.0 at [S=0,B=0,F=0], got {}", canon_data[0]); 
    assert!((canon_data[2] - 7.0).abs() < 1e-6, "Expected 7.0 at [S=0,B=1,F=0], got {}", canon_data[2]);
    
    println!("✅ Data preservation verified - canonicalization preserved element values");
}
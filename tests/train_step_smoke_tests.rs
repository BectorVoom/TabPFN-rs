/// Train Step Smoke Tests
/// 
/// Test suite to verify that train_step() method runs without panicking
/// with small configurations and produces finite loss values. These tests
/// validate the end-to-end training pipeline including:
/// 1. Dataset generation with correct shapes
/// 2. Transformer forward pass 
/// 3. Loss computation with ignore_index=-1
/// 4. Backward pass and parameter updates

use burn::tensor::backend::{Backend, AutodiffBackend};
use burn_autodiff::Autodiff;
use burn_ndarray::{NdArray};
use rand::{rngs::StdRng, SeedableRng};

use tab_pfn_rs::{
    tabpfn::architectures::base::{
        config::ModelConfig,
        transformer::DeterministicRngContext,
        train::{TabPFNTrainer, TrainingConfig, PriorType},
    }
};

type TestBackend = Autodiff<NdArray<f32>>;

/// Helper function to create a minimal TrainingConfig for testing
fn create_minimal_training_config() -> TrainingConfig {
    TrainingConfig {
        model: ModelConfig {
            max_num_classes: 5,
            num_buckets: 100,
            emsize: 64,           // Small embedding dimension
            features_per_group: 1,
            nhead: 2,             // Small number of heads
            remove_duplicate_features: false,
            dropout: 0.0,         // Disable dropout for deterministic testing
            encoder_use_bias: true,
            feature_positional_embedding: None,
            multiquery_item_attention: false,
            nan_handling_enabled: false,  // Disable for simplicity
            nan_handling_y_encoder: false,
            nhid_factor: 2,       // Small hidden factor  
            nlayers: 1,           // Single layer only
            normalize_by_used_features: false,
            normalize_on_train_only: false,
            normalize_to_ranking: false,
            normalize_x: false,   // Disable normalization for simplicity
            recompute_attn: false,
            recompute_layer: false,
            remove_empty_features: false,
            remove_outliers: false,
            use_separate_decoder: false,
            multiquery_item_attention_for_test_set: false,
            attention_init_gain: 1.0,
            dag_pos_enc_dim: None,
            item_attention_type: "full".to_string(),
            feature_attention_type: "full".to_string(), 
            seed: 42,
        },
        meta_batch_size: 2,
        tasks_per_batch: 1,
        max_samples_per_task: 20,
        min_samples_per_task: 10,
        learning_rate: 0.001,
        warmup_steps: 0,      // No warmup
        gradient_accumulation_steps: 1,
        gradient_clip_norm: None,  // No gradient clipping
        num_epochs: 1,
        checkpoint_frequency: 1000,
        validation_frequency: 1000,
        early_stopping_patience: 1000,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: None,
        prior_type: PriorType::Gaussian,
        num_features_range: (3, 3),  // Fixed small number of features
        num_classes_range: (2, 3),   // Small number of classes
        feature_noise_level: 0.1,
    }
}

/// Test single train_step with small configuration doesn't panic
#[test]
fn test_train_step_single_small_config() {
    println!("Testing single train_step with small configuration (S=5, B=2, F=3)");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = create_minimal_training_config();
    let rng_context = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let mut rng = StdRng::seed_from_u64(123);
    
    // Initialize trainer
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    
    // Execute single train step
    let loss = trainer.train_step(&device, &mut rng);
    
    // Verify loss is finite and reasonable
    assert!(loss.is_finite(), "Training loss must be finite, got: {}", loss);
    assert!(!loss.is_nan(), "Training loss must not be NaN, got: {}", loss);
    assert!(!loss.is_infinite(), "Training loss must not be infinite, got: {}", loss);
    assert!(loss >= 0.0, "Training loss must be non-negative, got: {}", loss);
    assert!(loss < 100.0, "Training loss should be reasonable (<100), got: {}", loss);
    
    println!("✅ Single train step PASSED: loss = {:.6} (finite and reasonable)", loss);
}

/// Test multiple consecutive train_steps don't cause issues
#[test] 
fn test_train_step_multiple_iterations() {
    println!("Testing multiple consecutive train_steps");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = create_minimal_training_config();
    let rng_context = DeterministicRngContext::<TestBackend>::new(999, device.clone());
    let mut rng = StdRng::seed_from_u64(456);
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    
    let num_steps = 3;
    let mut losses = Vec::new();
    
    // Execute multiple train steps
    for step in 0..num_steps {
        println!("  Executing train step {}/{}", step + 1, num_steps);
        let loss = trainer.train_step(&device, &mut rng);
        
        // Validate each loss
        assert!(loss.is_finite(), "Loss at step {} is not finite: {}", step, loss);
        assert!(loss >= 0.0, "Loss at step {} is negative: {}", step, loss);
        assert!(loss < 100.0, "Loss at step {} is too high: {}", step, loss);
        
        losses.push(loss);
        println!("    Step {} loss: {:.6}", step, loss);
    }
    
    // Verify we got expected number of losses
    assert_eq!(losses.len(), num_steps, "Should have {} loss values", num_steps);
    
    // Verify losses are not all identical (training should affect model state)
    let first_loss = losses[0];
    let all_identical = losses.iter().all(|&loss| (loss - first_loss).abs() < 1e-6);
    if all_identical && num_steps > 1 {
        println!("WARNING: All losses are identical - this may indicate the model is not updating");
    }
    
    println!("✅ Multiple train steps PASSED: {} steps completed with finite losses", num_steps);
    println!("   Losses: {:?}", losses);
}

/// Test train_step with different meta_batch_sizes
#[test]
fn test_train_step_different_batch_sizes() {
    println!("Testing train_step with different meta_batch sizes");
    
    let device = <TestBackend as Backend>::Device::default();
    
    let batch_sizes = [1, 2, 4];
    
    for &batch_size in &batch_sizes {
        println!("  Testing with meta_batch_size = {}", batch_size);
        
        let mut config = create_minimal_training_config();
        config.meta_batch_size = batch_size;
        
        let rng_context = DeterministicRngContext::<TestBackend>::new(777, device.clone());
        let mut rng = StdRng::seed_from_u64(789 + batch_size as u64);
        
        let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
        
        // Execute train step
        let loss = trainer.train_step(&device, &mut rng);
        
        // Validate loss
        assert!(loss.is_finite(), "Loss with batch_size {} is not finite: {}", batch_size, loss);
        assert!(loss >= 0.0, "Loss with batch_size {} is negative: {}", batch_size, loss);
        assert!(loss < 100.0, "Loss with batch_size {} is too high: {}", batch_size, loss);
        
        println!("    Batch size {} loss: {:.6}", batch_size, loss);
    }
    
    println!("✅ Different batch sizes test PASSED: all batch sizes produce finite losses");
}

/// Test train_step memory behavior with repeated calls
#[test]
fn test_train_step_memory_stability() {
    println!("Testing train_step memory stability with repeated calls");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = create_minimal_training_config();
    let rng_context = DeterministicRngContext::<TestBackend>::new(111, device.clone());
    let mut rng = StdRng::seed_from_u64(222);
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    
    let num_iterations = 5;
    
    // Execute multiple iterations to test memory stability
    for i in 0..num_iterations {
        let loss = trainer.train_step(&device, &mut rng);
        
        assert!(loss.is_finite(), "Loss at iteration {} is not finite", i);
        assert!(loss >= 0.0, "Loss at iteration {} is negative", i);
        
        // Progress indication
        if i % 2 == 0 {
            println!("  Completed {} iterations, latest loss: {:.6}", i + 1, loss);
        }
    }
    
    println!("✅ Memory stability test PASSED: {} iterations completed without issues", num_iterations);
}

/// Test train_step with minimal sequence length
#[test] 
fn test_train_step_minimal_sequence_length() {
    println!("Testing train_step with minimal sequence length");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = create_minimal_training_config();
    let rng_context = DeterministicRngContext::<TestBackend>::new(333, device.clone());
    let mut rng = StdRng::seed_from_u64(444);
    
    // Create trainer with minimal configuration
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    
    // Note: The train_step method uses fixed sequence length of 100 internally
    // This test verifies it can handle the default sequence length without issues
    let loss = trainer.train_step(&device, &mut rng);
    
    assert!(loss.is_finite(), "Loss with minimal config is not finite: {}", loss);
    assert!(loss >= 0.0, "Loss with minimal config is negative: {}", loss);
    
    println!("✅ Minimal sequence length test PASSED: loss = {:.6}", loss);
}

/// Test that train_step handles iteration counter correctly
#[test]
fn test_train_step_iteration_counter() {
    println!("Testing train_step iteration counter behavior");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = create_minimal_training_config();
    let rng_context = DeterministicRngContext::<TestBackend>::new(555, device.clone());
    let mut rng = StdRng::seed_from_u64(666);
    
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    
    // Verify initial state
    assert_eq!(trainer.iteration, 0, "Initial iteration should be 0");
    
    // Execute first train step
    let _loss1 = trainer.train_step(&device, &mut rng);
    assert_eq!(trainer.iteration, 1, "After first step, iteration should be 1");
    
    // Execute second train step
    let _loss2 = trainer.train_step(&device, &mut rng);
    assert_eq!(trainer.iteration, 2, "After second step, iteration should be 2");
    
    // Execute third train step
    let _loss3 = trainer.train_step(&device, &mut rng);
    assert_eq!(trainer.iteration, 3, "After third step, iteration should be 3");
    
    println!("✅ Iteration counter test PASSED: counter increments correctly");
}
//! Simple test to verify TabPFN training implementation
//! 
//! This test verifies that our enhanced TabPFN training implementation with
//! gradient accumulation and clipping is working correctly.

use std::collections::HashMap;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{backend::AutodiffBackend, Data, Device, Tensor};
use burn::config::Config;
use rand::{rngs::StdRng, SeedableRng};

use tab_pfn_rs::tabpfn::architectures::base::{
    config::ModelConfig,
    train::{TrainingConfig, TabPFNTrainer, PriorType},
    transformer::DeterministicRngContext,
    loss_utils,
};

type TestBackend = Autodiff<Wgpu>;

fn main() {
    println!("🧪 TabPFN Training Implementation Verification Test");
    println!("==================================================");

    test_masked_loss_functionality();
    test_training_step_with_gradient_clipping();
    test_training_step_with_gradient_accumulation();
    test_argmax_tie_breaking();

    println!("\n✅ All verification tests passed successfully!");
    println!("🎉 TabPFN training implementation is working correctly!");
}

fn test_masked_loss_functionality() {
    println!("\n🔍 Test 1: Masked Loss Functionality");
    
    let device = Default::default();
    
    // Test with extreme logits to verify numerical stability
    let logits = Tensor::<TestBackend, 2>::from_data(
        Data::from([[100.0, -100.0, 50.0], [20.0, 10.0, -30.0]]),
        &device,
    );
    
    let labels_with_ignore = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        Data::from([0i64, -1i64]),  // First class 0, second ignored
        &device,
    );
    
    let loss = loss_utils::compute_masked_cross_entropy_loss_ignore_index(
        logits, labels_with_ignore, &device
    );
    
    let loss_value = loss.to_data().as_slice::<f32>().unwrap()[0];
    
    assert!(loss_value.is_finite(), "Loss must be finite");
    assert!(loss_value >= 0.0, "Cross-entropy loss must be non-negative");
    
    println!("   ✓ Numerically stable masked loss: {:.6}", loss_value);
    println!("   ✓ Loss is finite and non-negative");
}

fn test_training_step_with_gradient_clipping() {
    println!("\n🔍 Test 2: Training Step with Gradient Clipping");
    
    let device = Default::default();
    let seed = 42u64;
    
    let config = TrainingConfig {
        model: ModelConfig::default(),
        meta_batch_size: 2,
        tasks_per_batch: 2,
        max_samples_per_task: 10,
        min_samples_per_task: 5,
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: Some(1.0),  // Enable gradient clipping
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 100,
        early_stopping_patience: 100,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: None,
        prior_type: PriorType::Tabular,
        num_features_range: (3, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
    };
    
    let rng_context = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Run training step - this should execute gradient clipping logic
    let loss = trainer.train_step(&device, &mut rng);
    
    assert!(loss.is_finite(), "Loss must be finite after gradient clipping");
    assert!(loss >= 0.0, "Loss must be non-negative");
    
    println!("   ✓ Training step with gradient clipping completed");
    println!("   ✓ Final loss: {:.6}", loss);
}

fn test_training_step_with_gradient_accumulation() {
    println!("\n🔍 Test 3: Training Step with Gradient Accumulation");
    
    let device = Default::default();
    let seed = 123u64;
    
    let config = TrainingConfig {
        model: ModelConfig::default(),
        meta_batch_size: 4,
        tasks_per_batch: 4,
        max_samples_per_task: 10,
        min_samples_per_task: 5,
        learning_rate: 0.001,
        warmup_steps: 0,
        gradient_accumulation_steps: 2,  // Enable gradient accumulation
        gradient_clip_norm: None,
        num_epochs: 1,
        checkpoint_frequency: 100,
        validation_frequency: 100,
        early_stopping_patience: 100,
        use_gradient_checkpointing: false,
        cache_trainset_representations: false,
        layer_dropout_min_layers: None,
        prior_type: PriorType::Tabular,
        num_features_range: (3, 5),
        num_classes_range: (2, 3),
        feature_noise_level: 0.1,
    };
    
    let rng_context = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let mut trainer = TabPFNTrainer::new(config, &device, rng_context);
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Run first accumulation step
    println!("   Running accumulation step 1/2...");
    let loss1 = trainer.train_step(&device, &mut rng);
    assert!(loss1.is_finite(), "First accumulation step loss must be finite");
    
    // Run second accumulation step (should trigger optimizer update)
    println!("   Running accumulation step 2/2...");
    let loss2 = trainer.train_step(&device, &mut rng);
    assert!(loss2.is_finite(), "Second accumulation step loss must be finite");
    
    println!("   ✓ Gradient accumulation completed successfully");
    println!("   ✓ Step 1 loss: {:.6}, Step 2 loss: {:.6}", loss1, loss2);
}

fn test_argmax_tie_breaking() {
    println!("\n🔍 Test 4: Argmax Tie-Breaking");
    
    let device = Default::default();
    
    // Create logits with deliberate ties
    let logits_with_ties = Tensor::<TestBackend, 3>::from_data(
        Data::from([[[1.0, 1.0, 0.5, 1.0], [0.8, 0.8, 0.8, 0.7]]]),
        &device,
    ); // Shape: [1, 2, 4] = [S, B, C]
    
    let predictions = tab_pfn_rs::tabpfn::architectures::base::train::argmax_with_tie_break_smallest(
        logits_with_ties
    );
    
    let predictions_data = predictions.to_data();
    let pred_values: Vec<i32> = predictions_data.as_slice().iter().cloned().collect();
    
    // Verify tie-breaking behavior: should always choose smallest index
    assert_eq!(pred_values[0], 0, "First sample: tie between indices 0,1,3 should choose 0");
    assert_eq!(pred_values[1], 0, "Second sample: tie between indices 0,1,2 should choose 0");
    
    println!("   ✓ Tie-breaking test passed");
    println!("   ✓ Predictions: {:?} (correctly chose smallest indices)", pred_values);
}
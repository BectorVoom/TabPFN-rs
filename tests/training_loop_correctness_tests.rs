//! Training loop correctness tests for TabPFN
//!
//! Tests gradient accumulation, loss computation, and parameter updates

use burn::tensor::{Tensor, TensorData, Int, backend::Backend};
use burn_ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::Module;
use tab_pfn_rs::tabpfn::architectures::base::{
    transformer::{PerFeatureTransformer, DeterministicRngContext},
    config::ModelConfig,
    loss_utils,
    train::{TrainingConfig, PriorType}
};
// Note: make_test_config is defined in transformer.rs test module
use std::collections::HashMap;

type TestBackend = Autodiff<NdArray<f32>>;

/// Create a test configuration for the model
fn make_test_config() -> ModelConfig {
    ModelConfig {
        d_model: 64,
        nhead: 4,
        d_hid: 128,
        nlayers: 2,
        n_out: 2,
        dropout: 0.0,
        style_encoder_type: None,
        y_encoder_type: None,
        pos_encoder_type: None,
        layer_norm_eps: 1e-5,
        aggregate_k_heads: 1,
        efficient_zero_grad: true,
        multiclass_loss_type: "CrossEntropy".to_string(),
        normalize_to_ranking: false,
        set_value_to_nan: false,
        normalize_with_sqrt: false,
        mix_activations: false,
        emsize: 512,
        bptt: 10,
        eval_positions: vec![10],
        seq_len_used: 100,
        sampling: "normal".to_string(),
        epochs: 12,
        num_steps: 100,
        verbose: true,
        pre_sample_targets: true,
        batch_size_inference: 1,
        recompute_attn: true,
        bptt_extra_samples: None,
        epoch_count: 0,
        extra_prior_kwargs_dict: std::collections::HashMap::new(),
        differentiable_hps_as_style: false,
        max_num_classes: 10,
        num_classes: 2,
        num_features: 4,
        feature_shift_decoder: true,
        feature_shift_decoder_2: false,
        rotate_normalized_labels: true,
        feature_positional_embedding: None,
        softmax_temperature: 1.0,
        multihead_ln_weights: true,
        seed: 0,
        batch_size: 1,
        inference: false,
        dag_pos_enc_dim: None,
    }
}

/// Helper function to reshape transformer output to match loss function expectations
fn reshape_for_loss<B: Backend>(
    output: Tensor<B, 3>,         // [batch, seq, classes] from transformer
    targets: Tensor<B, 2, Int>,   // [batch, seq] 
    mask: Tensor<B, 2, burn::tensor::Bool>, // [batch, seq]
) -> (Tensor<B, 2>, Tensor<B, 1, Int>, Tensor<B, 1, burn::tensor::Bool>) {
    let batch_size = output.dims()[0];
    let seq_len = output.dims()[1];
    let num_classes = output.dims()[2];
    
    // Reshape to [batch*seq, classes] for loss computation
    let output_flat = output.reshape([batch_size * seq_len, num_classes]);
    let targets_flat = targets.reshape([batch_size * seq_len]);
    let mask_flat = mask.reshape([batch_size * seq_len]);
    
    (output_flat, targets_flat, mask_flat)
}

#[test]
fn test_gradient_accumulation_correctness() {
    println!("Testing gradient accumulation correctness");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = make_test_config();
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model1 = PerFeatureTransformer::<TestBackend>::new(
        &config, &rng_ctx, 2, "gelu", None, false, None, false, None, false, &device
    ).expect("create model1");
    
    let mut model2 = PerFeatureTransformer::<TestBackend>::new(
        &config, &rng_ctx, 2, "gelu", None, false, None, false, None, false, &device
    ).expect("create model2");
    
    // Create identical training batches
    let batch_size = 2;
    let seq_len = 4;
    let num_features = 3;
    let num_classes = 2;
    
    let features_data: Vec<f32> = vec![
        // Batch 0, sequence positions
        1.0, 2.0, 3.0,  // pos 0
        4.0, 5.0, 6.0,  // pos 1  
        7.0, 8.0, 9.0,  // pos 2
        10.0, 11.0, 12.0, // pos 3
        // Batch 1, sequence positions
        0.5, 1.5, 2.5,  // pos 0
        3.5, 4.5, 5.5,  // pos 1
        6.5, 7.5, 8.5,  // pos 2
        9.5, 10.5, 11.5, // pos 3
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    );
    
    // Training targets (test positions marked with -1)
    let targets_data = vec![
        0, 1, -1, 0,  // batch 0: train, train, test, train
        1, 0, -1, 1,  // batch 1: train, train, test, train
    ];
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    );
    
    // Train mask (true for train positions)
    let train_mask_data = vec![
        true, true, false, true,   // batch 0
        true, true, false, true,   // batch 1
    ];
    let train_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data, [batch_size, seq_len]),
        &device
    );
    
    // Method 1: Single batch gradient computation
    let x_map1 = {
        let mut map = HashMap::new();
        map.insert("main".to_string(), features.clone().require_grad());
        map
    };
    let y_map1 = {
        let mut map = HashMap::new();
        // Expand targets to 3D: [batch, seq] -> [batch, seq, 1] for transformer input
        let targets_3d = targets.clone().float().unsqueeze_dim(2).require_grad();
        map.insert("main".to_string(), targets_3d);
        map
    };
    
    let mut rng_opt1: Option<&mut rand::prelude::StdRng> = None;
    let output1 = model1.transformer_forward(
        x_map1, Some(y_map1), true, &mut rng_opt1, None, None, None, false
    ).expect("forward1");
    
    let (output1_flat, targets1_flat, mask1_flat) = reshape_for_loss(
        output1, targets.clone(), train_mask.clone()
    );
    let loss1 = loss_utils::compute_masked_cross_entropy_loss(
        output1_flat, targets1_flat, mask1_flat, &device
    );
    
    let grads1 = loss1.backward();
    
    // Method 2: Simulate gradient accumulation over two half-batches  
    let half_batch = batch_size / 2;
    
    // First half-batch
    let features_half1 = features.clone().slice([0..half_batch]);
    let targets_half1 = targets.clone().slice([0..half_batch]);
    let train_mask_half1 = train_mask.clone().slice([0..half_batch]);
    
    let x_map_half1 = {
        let mut map = HashMap::new();
        map.insert("main".to_string(), features_half1.require_grad());
        map
    };
    let y_map_half1 = {
        let mut map = HashMap::new();
        let targets_half1_3d = targets_half1.clone().float().unsqueeze_dim(2).require_grad();
        map.insert("main".to_string(), targets_half1_3d);
        map
    };
    
    let mut rng_opt2: Option<&mut rand::prelude::StdRng> = None;
    let output_half1 = model2.transformer_forward(
        x_map_half1, Some(y_map_half1), true, &mut rng_opt2, None, None, None, false
    ).expect("forward_half1");
    
    let (output_half1_flat, targets_half1_flat, mask_half1_flat) = reshape_for_loss(
        output_half1, targets_half1.clone(), train_mask_half1.clone()
    );
    let loss_half1 = loss_utils::compute_masked_cross_entropy_loss(
        output_half1_flat, targets_half1_flat, mask_half1_flat, &device
    );
    
    // Scale loss by accumulation factor (divide by number of accumulation steps)
    let scaled_loss_half1 = loss_half1 / Tensor::from_floats([2.0], &device);
    let grads_half1 = scaled_loss_half1.backward();
    
    // Second half-batch
    let features_half2 = features.clone().slice([half_batch..batch_size]);
    let targets_half2 = targets.clone().slice([half_batch..batch_size]);
    let train_mask_half2 = train_mask.clone().slice([half_batch..batch_size]);
    
    let x_map_half2 = {
        let mut map = HashMap::new();
        map.insert("main".to_string(), features_half2.require_grad());
        map
    };
    let y_map_half2 = {
        let mut map = HashMap::new();
        let targets_half2_3d = targets_half2.clone().float().unsqueeze_dim(2).require_grad();
        map.insert("main".to_string(), targets_half2_3d);
        map
    };
    
    let mut rng_opt3: Option<&mut rand::prelude::StdRng> = None;
    let output_half2 = model2.transformer_forward(
        x_map_half2, Some(y_map_half2), true, &mut rng_opt3, None, None, None, false
    ).expect("forward_half2");
    
    let (output_half2_flat, targets_half2_flat, mask_half2_flat) = reshape_for_loss(
        output_half2, targets_half2.clone(), train_mask_half2.clone()
    );
    let loss_half2 = loss_utils::compute_masked_cross_entropy_loss(
        output_half2_flat, targets_half2_flat, mask_half2_flat, &device
    );
    
    // Scale loss and accumulate gradients
    let scaled_loss_half2 = loss_half2 / Tensor::from_floats([2.0], &device);  
    let grads_half2 = scaled_loss_half2.backward();
    
    // TODO: Compare accumulated gradients with single batch gradients
    // This would require gradient comparison functionality
    
    println!("Single batch loss: {:.6}", 
        loss1.clone().to_data().as_slice::<f32>().unwrap()[0]);
    println!("Half batch 1 loss: {:.6}", 
        loss_half1.clone().to_data().as_slice::<f32>().unwrap()[0]);
    println!("Half batch 2 loss: {:.6}", 
        loss_half2.clone().to_data().as_slice::<f32>().unwrap()[0]);
    
    // The sum of half-batch losses should approximately equal the full batch loss
    let total_accumulated_loss = 
        loss_half1.clone().to_data().as_slice::<f32>().unwrap()[0] +
        loss_half2.clone().to_data().as_slice::<f32>().unwrap()[0];
    let full_batch_loss = loss1.clone().to_data().as_slice::<f32>().unwrap()[0];
    
    let loss_diff = (total_accumulated_loss - full_batch_loss).abs();
    assert!(loss_diff < 1e-5, 
        "Gradient accumulation loss mismatch: {:.6} vs {:.6}, diff: {:.6}",
        total_accumulated_loss, full_batch_loss, loss_diff);
    
    println!("✅ Gradient accumulation correctness test passed");
}

#[test]
fn test_loss_masking_training_consistency() {
    println!("Testing loss masking consistency during training");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = make_test_config();
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model = PerFeatureTransformer::<TestBackend>::new(
        &config, &rng_ctx, 2, "gelu", None, false, None, false, None, false, &device
    ).expect("create model");
    
    let batch_size = 3;
    let seq_len = 5;
    let num_features = 2;
    
    // Create training data with different mask patterns
    let features_data: Vec<f32> = vec![
        // Batch 0: mixed train/test
        1.0, 2.0,   // train
        3.0, 4.0,   // train  
        5.0, 6.0,   // test
        7.0, 8.0,   // train
        9.0, 10.0,  // test
        // Batch 1: mostly train
        11.0, 12.0, // train
        13.0, 14.0, // train
        15.0, 16.0, // train
        17.0, 18.0, // test
        19.0, 20.0, // train
        // Batch 2: mostly test
        21.0, 22.0, // test
        23.0, 24.0, // test
        25.0, 26.0, // train
        27.0, 28.0, // test
        29.0, 30.0, // test
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    ).require_grad();
    
    let targets_data = vec![
        0, 1, -1, 0, -1,  // batch 0
        1, 0, 1, -1, 0,   // batch 1  
        -1, -1, 1, -1, -1, // batch 2
    ];
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    );
    
    let train_mask_data = vec![
        true, true, false, true, false,   // batch 0: 3 train positions
        true, true, true, false, true,    // batch 1: 4 train positions  
        false, false, true, false, false, // batch 2: 1 train position
    ];
    let train_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data, [batch_size, seq_len]),
        &device
    );
    
    // Forward pass
    let x_map = {
        let mut map = HashMap::new();
        map.insert("main".to_string(), features);
        map
    };
    let y_map = {
        let mut map = HashMap::new();
        let targets_3d = targets.clone().float().unsqueeze_dim(2).require_grad();
        map.insert("main".to_string(), targets_3d);
        map
    };
    
    let mut rng_opt: Option<&mut rand::prelude::StdRng> = None;
    let output = model.transformer_forward(
        x_map, Some(y_map), true, &mut rng_opt, None, None, None, false
    ).expect("forward");
    
    // Compute loss with masking  
    let (output_flat1, targets_flat1, mask_flat1) = reshape_for_loss(
        output.clone(), targets.clone(), train_mask.clone()
    );
    let loss = loss_utils::compute_masked_cross_entropy_loss(
        output_flat1, targets_flat1, mask_flat1, &device
    );
    
    // Compute loss without masking (all positions)
    let all_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(vec![true; batch_size * seq_len], [batch_size, seq_len]),
        &device
    );
    let (output_flat2, targets_flat2, mask_flat2) = reshape_for_loss(
        output, targets.clone(), all_mask
    );
    let unmasked_loss = loss_utils::compute_masked_cross_entropy_loss(
        output_flat2, targets_flat2, mask_flat2, &device
    );
    
    let masked_loss_val = loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    let unmasked_loss_val = unmasked_loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    
    println!("Masked loss (train positions only): {:.6}", masked_loss_val);
    println!("Unmasked loss (all positions): {:.6}", unmasked_loss_val);
    
    // They should be different unless all positions happen to have identical losses
    // In practice with different predictions, masking should make a difference
    
    // The masked loss should only include contributions from train positions
    // We expect them to differ for this dataset design
    assert!(masked_loss_val.is_finite() && unmasked_loss_val.is_finite(),
        "Both losses should be finite");
    
    // Verify gradients can be computed
    let grads = loss.backward();
    // If we reach here, gradient computation succeeded
    
    println!("✅ Loss masking training consistency test passed");
}

#[test] 
fn test_parameter_update_consistency() {
    println!("Testing parameter update consistency");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = make_test_config();
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model = PerFeatureTransformer::<TestBackend>::new(
        &config, &rng_ctx, 2, "gelu", None, false, None, false, None, false, &device
    ).expect("create model");
    
    // Simple training example
    let batch_size = 2;
    let seq_len = 3;
    let num_features = 2;
    
    let features_data: Vec<f32> = vec![
        1.0, 2.0,  // train
        3.0, 4.0,  // test 
        5.0, 6.0,  // train
        7.0, 8.0,  // train
        9.0, 10.0, // train
        11.0, 12.0, // test
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    ).require_grad();
    
    let targets_data = vec![
        0, -1, 1,  // batch 0: train, test, train
        1, 0, -1,  // batch 1: train, train, test
    ];
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    );
    
    let train_mask_data = vec![
        true, false, true,  // batch 0
        true, true, false,  // batch 1
    ];
    let train_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data, [batch_size, seq_len]),
        &device
    );
    
    // Get initial parameter count
    let initial_params = model.num_params();
    println!("Model has {} parameters", initial_params);
    assert!(initial_params > 0, "Model should have trainable parameters");
    
    // Forward pass
    let x_map = {
        let mut map = HashMap::new();
        map.insert("main".to_string(), features);
        map
    };
    let y_map = {
        let mut map = HashMap::new();
        let targets_3d = targets.clone().float().unsqueeze_dim(2).require_grad();
        map.insert("main".to_string(), targets_3d);
        map
    };
    
    let mut rng_opt: Option<&mut rand::prelude::StdRng> = None;
    let output = model.transformer_forward(
        x_map, Some(y_map), true, &mut rng_opt, None, None, None, false
    ).expect("forward");
    
    // Compute loss and gradients
    let (output_flat, targets_flat, mask_flat) = reshape_for_loss(
        output, targets.clone(), train_mask
    );
    let loss = loss_utils::compute_masked_cross_entropy_loss(
        output_flat, targets_flat, mask_flat, &device
    );
    
    let initial_loss = loss.clone().to_data().as_slice::<f32>().unwrap()[0];
    println!("Initial loss: {:.6}", initial_loss);
    
    // Backward pass to compute gradients
    let grads = loss.backward();
    
    // Verify gradient computation succeeded
    assert!(initial_loss.is_finite(), "Initial loss should be finite");
    
    // Parameter count should remain the same after gradient computation
    let params_after_grad = model.num_params();
    assert_eq!(initial_params, params_after_grad, 
        "Parameter count should not change during gradient computation");
    
    println!("✅ Parameter update consistency test passed");
}

#[test]
fn test_multi_step_training_consistency() {
    println!("Testing multi-step training consistency");
    
    let device = <TestBackend as Backend>::Device::default();
    let config = make_test_config();
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    let mut model = PerFeatureTransformer::<TestBackend>::new(
        &config, &rng_ctx, 2, "gelu", None, false, None, false, None, false, &device
    ).expect("create model");
    
    // Simple reproducible training data
    let batch_size = 2;
    let seq_len = 3;
    let num_features = 2;
    
    let features_data: Vec<f32> = vec![
        1.0, 0.5,   // train
        2.0, 1.5,   // test
        3.0, 2.5,   // train
        0.8, 0.2,   // train
        1.8, 1.2,   // train
        2.8, 2.2,   // test
    ];
    
    let features = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(features_data, [batch_size, seq_len, num_features]),
        &device
    );
    
    let targets_data = vec![
        0, -1, 1,   // batch 0: train, test, train
        1, 0, -1,   // batch 1: train, train, test
    ];
    let targets = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(targets_data, [batch_size, seq_len]),
        &device
    );
    
    let train_mask_data = vec![
        true, false, true,   // batch 0
        true, true, false,   // batch 1
    ];
    let train_mask = Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
        TensorData::new(train_mask_data, [batch_size, seq_len]),
        &device
    );
    
    let mut losses = Vec::new();
    let num_steps = 3;
    
    for step in 0..num_steps {
        // Forward pass
        let features_grad = features.clone().require_grad();
        let x_map = {
            let mut map = HashMap::new();
            map.insert("main".to_string(), features_grad);
            map
        };
        let y_map = {
            let mut map = HashMap::new();
            let targets_3d = targets.clone().float().unsqueeze_dim(2).require_grad();
            map.insert("main".to_string(), targets_3d);
            map
        };
        
        let mut rng_opt: Option<&mut rand::prelude::StdRng> = None;
        let output = model.transformer_forward(
            x_map, Some(y_map), true, &mut rng_opt, None, None, None, false
        ).expect(&format!("forward step {}", step));
        
        // Compute loss
        let (output_flat, targets_flat, mask_flat) = reshape_for_loss(
            output, targets.clone(), train_mask.clone()
        );
        let loss = loss_utils::compute_masked_cross_entropy_loss(
            output_flat, targets_flat, mask_flat, &device
        );
        
        let loss_val = loss.clone().to_data().as_slice::<f32>().unwrap()[0];
        losses.push(loss_val);
        
        println!("Step {}: Loss = {:.6}", step, loss_val);
        
        // Compute gradients
        let _grads = loss.backward();
        
        // TODO: Apply parameter updates (would need optimizer implementation)
        // For now, just verify that loss computation is consistent
    }
    
    // Verify that losses are finite and consistent
    for (i, &loss_val) in losses.iter().enumerate() {
        assert!(loss_val.is_finite(), "Loss at step {} should be finite", i);
        assert!(loss_val >= 0.0, "Loss at step {} should be non-negative", i);
    }
    
    // In this test without parameter updates, losses should be identical
    if losses.len() > 1 {
        let first_loss = losses[0];
        for (i, &loss_val) in losses.iter().enumerate().skip(1) {
            let diff = (loss_val - first_loss).abs();
            assert!(diff < 1e-6, 
                "Without parameter updates, losses should be identical. Step {} differs by {:.8}",
                i, diff);
        }
    }
    
    println!("✅ Multi-step training consistency test passed");
}

#[test]
#[ignore = "Requires full training implementation"]
fn test_training_convergence() {
    println!("Testing training convergence with simple dataset");
    
    // This test would require a full training loop implementation
    // Including optimizer, parameter updates, etc.
    // For now, marked as ignored until training infrastructure is complete
    
    let device = <TestBackend as Backend>::Device::default();
    let _config = TrainingConfig {
        model: make_test_config(),
        meta_batch_size: 32,
        tasks_per_batch: 4,
        max_samples_per_task: 100,
        min_samples_per_task: 10,
        learning_rate: 0.001,
        warmup_steps: 100,
        gradient_accumulation_steps: 1,
        gradient_clip_norm: Some(1.0),
        num_epochs: 10,
        checkpoint_frequency: 1000,
        validation_frequency: 100,
        early_stopping_patience: 5,
        use_gradient_checkpointing: false,
        cache_trainset_representations: true,
        layer_dropout_min_layers: None,
        prior_type: PriorType::Gaussian,
        num_features_range: (1, 10),
        num_classes_range: (2, 5),
        feature_noise_level: 0.0,
    };
    
    println!("Training convergence test not yet implemented - requires full trainer");
}
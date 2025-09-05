use burn::tensor::{backend::Backend, Tensor};
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::layer::LayerNorm;
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;

type TestBackend = NdArray<f32>;

/// Simple compilation test - just make sure LayerNorm can be created and called
#[test]
fn test_layernorm_basic_functionality() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Simple 3D input: (1, 2, 4)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([1, 2, 4]);

    // Create LayerNorm
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);

    // Apply LayerNorm
    let output = layer_norm.layernorm_forward_3d(input_tensor.clone(), true, None);

    // Verify basic properties
    assert_eq!(output.shape().dims, input_tensor.shape().dims, "Shape should be preserved");
    
    // Extract data to verify it contains reasonable values
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    assert_eq!(output_data.len(), 8, "Should have 8 output elements");
    
    // Just verify output is finite (not NaN or Inf)
    for (i, &value) in output_data.iter().enumerate() {
        assert!(value.is_finite(), "Output element {} should be finite, got {}", i, value);
    }

    println!("✅ Basic LayerNorm functionality test passed");
}

/// Test 4D tensor with the same golden fixture values reshaped
#[test]
fn test_layernorm_4d_golden_fixture() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Same data as 3D test but reshaped to (1, 2, 3, 4)
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        0.5, -1.0, 0.0, 2.0,
        10.0, 11.0, 12.0, 13.0,
        -1.0, -2.0, -3.0, -4.0,
        4.0, 0.0, -4.0, 2.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    
    let input_tensor: Tensor<TestBackend, 4> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([1, 2, 3, 4]);

    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);
    let output = layer_norm.layernorm_forward_4d(input_tensor, true, None);
    
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    
    // Same expected values as 3D test
    let expected: Vec<f32> = vec![
        -1.3416355, -0.44721183, 0.44721183, 1.3416355,
        0.11546957, -1.2701652, -0.3464087, 1.5011044,
        -1.3416355, -0.44721183, 0.44721183, 1.3416355,
        1.3416355, 0.44721183, -0.44721183, -1.3416355,
        1.1832154, -0.16903076, -1.5212768, 0.5070923,
        0.0, 0.0, 0.0, 0.0,
    ];

    let tolerance = 1e-5f32;
    for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
        let diff = (actual as f32 - expected_val as f32).abs();
        assert!(
            diff <= tolerance,
            "4D Element {} differs: got {}, expected {}, diff {}",
            i, actual, expected_val, diff
        );
    }
}

/// Test shape preservation - output shape must match input shape exactly
#[test]
fn test_layernorm_shape_preservation() {
    let device = <TestBackend as Backend>::Device::default();
    
    let test_cases = vec![
        ([2, 3, 4], vec![4]),      // Original fixture shape
        ([1, 5, 4], vec![4]),      // Single batch
        ([3, 1, 4], vec![4]),      // Single sequence item
        ([1, 1, 4], vec![4]),      // Minimal shape
        ([2, 4, 8], vec![8]),      // Different feature dimension
    ];
    
    for (input_shape, normalized_shape) in test_cases {
        let total_elements = input_shape.iter().product();
        let input_data: Vec<f32> = (0..total_elements).map(|i| i as f32 * 0.1).collect();
        
        let input_tensor: Tensor<TestBackend, 3> = 
            Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
            .reshape(input_shape);
        
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
        let layer_norm = LayerNorm::new(normalized_shape, 1e-5, true, &rng_ctx);
        let output = layer_norm.layernorm_forward_3d(input_tensor.clone(), true, None);
        
        assert_eq!(
            output.shape().dims,
            input_tensor.shape().dims,
            "Shape not preserved for input shape {:?}",
            input_shape
        );
    }
}

/// Negative test: reshape divisibility mismatch should panic with clear message
#[test]
#[should_panic(expected = "LayerNorm shape mismatch")]
fn test_layernorm_reshape_mismatch_error() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Input shape (2, 3, 5) but normalized_shape=[4] - not divisible
    let input_data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([2, 3, 5]);

    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);
    
    // This should panic with "LayerNorm shape mismatch" message
    let _output = layer_norm.layernorm_forward_3d(input_tensor, true, None);
}

/// Test mathematical properties of LayerNorm: zero mean and unit variance
#[test]
fn test_layernorm_mathematical_properties() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Create input with various ranges
    let input_data: Vec<f32> = vec![
        10.0, 20.0, 30.0, 40.0,    // Large positive values
        -5.0, -10.0, 5.0, 15.0,    // Mixed signs
        0.1, 0.2, 0.3, 0.4,        // Small positive values
    ];
    
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([3, 1, 4]);

    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);
    let output = layer_norm.layernorm_forward_3d(input_tensor, true, None);
    
    let output_data = output.into_data().to_vec().unwrap();
    
    // Check normalization properties for each normalized group (every 4 elements)
    for chunk_idx in 0..(output_data.len() / 4) {
        let chunk = &output_data[chunk_idx * 4..(chunk_idx + 1) * 4];
        
        // Calculate mean and variance
        let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
        let variance: f32 = chunk.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / chunk.len() as f32;
        
        // LayerNorm should produce zero mean and unit variance (within numerical precision)
        assert!(
            mean.abs() < 1e-6,
            "Mean not close to zero for chunk {}: {}",
            chunk_idx, mean
        );
        assert!(
            (variance - 1.0).abs() < 1e-3,
            "Variance not close to 1.0 for chunk {}: {}",
            chunk_idx, variance
        );
    }
}

/// Test zero input handling (edge case)
#[test]
fn test_layernorm_zero_input() {
    let device = <TestBackend as Backend>::Device::default();
    
    // All-zero input should produce all-zero output
    let input_data: Vec<f32> = vec![0.0; 8]; // 2 x 1 x 4
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([2, 1, 4]);

    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);
    let output = layer_norm.layernorm_forward_3d(input_tensor, true, None);
    
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    
    // All outputs should be zero (or very close due to numerical precision)
    for (i, &value) in output_data.iter().enumerate() {
        assert!(
            value.abs() < 1e-6,
            "Zero input should produce zero output, but element {} = {}",
            i, value
        );
    }
}

/// Performance/stress test with larger tensors
#[test]
fn test_layernorm_large_tensor() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test with larger tensor: (8, 16, 64)
    let batch_size = 8;
    let seq_len = 16; 
    let feature_dim = 64;
    let total_elements = batch_size * seq_len * feature_dim;
    
    let input_data: Vec<f32> = (0..total_elements)
        .map(|i| (i as f32 * 0.001).sin()) // Varied input pattern
        .collect();
    
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([batch_size, seq_len, feature_dim]);

    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![feature_dim], 1e-5, true, &rng_ctx);
    let output = layer_norm.layernorm_forward_3d(input_tensor.clone(), true, None);
    
    // Verify shape preservation
    assert_eq!(output.shape().dims, input_tensor.shape().dims);
    
    // Spot check: verify normalization on a few random chunks
    let output_data = output.into_data().to_vec().unwrap();
    for chunk_start in [0, 1000, 5000] {
        if chunk_start + feature_dim <= output_data.len() {
            let chunk = &output_data[chunk_start..chunk_start + feature_dim];
            let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
            assert!(mean.abs() < 1e-2, "Large tensor normalization failed: mean = {}", mean);
        }
    }
}

/// Test tensor compatibility validation in LayerNorm
#[test]
fn test_layernorm_tensor_compatibility() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Create LayerNorm
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let layer_norm = LayerNorm::new(vec![4], 1e-5, true, &rng_ctx);
    
    // Test validation method (should always pass for NdArray backend)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_tensor: Tensor<TestBackend, 3> = 
        Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
        .reshape([1, 1, 4]);
    
    let validation_result = layer_norm.validate_tensor_compatibility(&input_tensor);
    assert!(validation_result.is_ok(), "Tensor compatibility validation should pass");
    
    println!("✅ Tensor compatibility test passed");
}

/// Test that device consistency is maintained in PerFeatureEncoderLayer
#[test]  
fn test_per_feature_encoder_device_consistency() {
    use tab_pfn_rs::tabpfn::architectures::base::layer::PerFeatureEncoderLayer;
    use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;
    
    let device = <TestBackend as Backend>::Device::default();
    
    // Create a minimal config for testing
    let mut config = ModelConfig::default();
    config.emsize = 64;
    config.nhead = 8;
    config.multiquery_item_attention = false;
    config.multiquery_item_attention_for_test_set = false;
    config.recompute_attn = false;
    
    // Create RNG context for deterministic initialization
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    
    // Test successful creation with consistent device usage
    let result = PerFeatureEncoderLayer::<TestBackend>::new(
        &config,
        256, // dim_feedforward
        "GELU".to_string(),
        1e-5, // layer_norm_eps
        false, // pre_norm
        &device,
        false, // second_mlp
        true, // layer_norm_with_elementwise_affine
        false, // zero_init
        None, // save_peak_mem_factor
        false, // attention_between_features
        None, // d_k
        None, // d_v
        None, // precomputed_kv
        &rng_ctx, // rng_ctx
        0, // seed_offset
    );
    
    assert!(result.is_ok(), "PerFeatureEncoderLayer creation should succeed with consistent device usage");
    
    println!("✅ PerFeatureEncoderLayer device consistency test passed");
}
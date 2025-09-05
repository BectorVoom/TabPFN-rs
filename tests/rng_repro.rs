// tests/rng_repro.rs - Test RNG reproducibility and determinism
use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn_ndarray::NdArray;
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;

type TestBackend = NdArray<f32>;

#[test]
fn test_deterministic_rng_context_creation() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that DeterministicRngContext can be created with same seed
    let seed = 42u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    // Both contexts should be created successfully
    // (Detailed equality testing would require exposing internal state)
    assert_eq!(format!("{:?}", rng_ctx1).len() > 0, true, "RNG context 1 should have debug representation");
    assert_eq!(format!("{:?}", rng_ctx2).len() > 0, true, "RNG context 2 should have debug representation");
}

#[test]
fn test_deterministic_rng_different_seeds() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test RNG contexts with different seeds
    let rng_ctx_42 = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let rng_ctx_123 = DeterministicRngContext::<TestBackend>::new(123, device.clone());
    
    // Should create different contexts (can't directly compare without exposing internals)
    // This test mainly ensures both contexts can be created without errors
    let debug1 = format!("{:?}", rng_ctx_42);
    let debug2 = format!("{:?}", rng_ctx_123);
    
    assert!(debug1.len() > 0, "RNG context with seed 42 should be valid");
    assert!(debug2.len() > 0, "RNG context with seed 123 should be valid");
}

#[test]
fn test_deterministic_linear_layer_reproducibility() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that deterministic linear layers produce identical weights with same seed
    let seed = 42u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    // Create linear layers with same RNG context
    let linear1 = rng_ctx1.create_deterministic_linear(3, 2, true, 1000);
    let linear2 = rng_ctx2.create_deterministic_linear(3, 2, true, 1000);
    
    // Extract weights and compare
    let weights1_data = linear1.weight.val().to_data();
    let weights2_data = linear2.weight.val().to_data();
    
    let weights1_values = weights1_data.as_slice::<f32>().unwrap();
    let weights2_values = weights2_data.as_slice::<f32>().unwrap();
    
    // Weights should be identical for same seed
    assert_eq!(weights1_values.len(), weights2_values.len(), "Weight tensor sizes should match");
    
    for (w1, w2) in weights1_values.iter().zip(weights2_values.iter()) {
        assert!((w1 - w2).abs() < 1e-8, 
                "Deterministic linear layers should produce identical weights: {} vs {}", w1, w2);
    }
    
    // Also check biases if they exist
    if let (Some(bias1), Some(bias2)) = (linear1.bias.as_ref(), linear2.bias.as_ref()) {
        let bias1_data = bias1.val().to_data();
        let bias2_data = bias2.val().to_data();
        
        let bias1_values = bias1_data.as_slice::<f32>().unwrap();
        let bias2_values = bias2_data.as_slice::<f32>().unwrap();
        
        for (b1, b2) in bias1_values.iter().zip(bias2_values.iter()) {
            assert!((b1 - b2).abs() < 1e-8,
                    "Deterministic linear biases should be identical: {} vs {}", b1, b2);
        }
    }
}

#[test]
fn test_deterministic_linear_different_seeds_produce_different_weights() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that different seeds produce different weights
    let rng_ctx_42 = DeterministicRngContext::<TestBackend>::new(42, device.clone());
    let rng_ctx_123 = DeterministicRngContext::<TestBackend>::new(123, device.clone());
    
    let linear_42 = rng_ctx_42.create_deterministic_linear(4, 3, true, 0);
    let linear_123 = rng_ctx_123.create_deterministic_linear(4, 3, true, 0);
    
    let weights_42_data = linear_42.weight.val().to_data();
    let weights_123_data = linear_123.weight.val().to_data();
    
    let weights_42 = weights_42_data.as_slice::<f32>().unwrap();
    let weights_123 = weights_123_data.as_slice::<f32>().unwrap();
    
    // At least some weights should be different
    let mut found_difference = false;
    for (w42, w123) in weights_42.iter().zip(weights_123.iter()) {
        if (w42 - w123).abs() > 1e-6 {
            found_difference = true;
            break;
        }
    }
    
    assert!(found_difference, "Different seeds should produce different weights");
}

#[test]
fn test_deterministic_layer_norm_reproducibility() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test LayerNorm reproducibility
    let seed = 12345u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    let layer_norm1 = rng_ctx1.create_deterministic_layer_norm(64, 1e-5);
    let layer_norm2 = rng_ctx2.create_deterministic_layer_norm(64, 1e-5);
    
    // LayerNorm weights should be identical (they're typically initialized to 1.0)
    let weight1_data = layer_norm1.weight.val().to_data();
    let weight2_data = layer_norm2.weight.val().to_data();
    
    let weight1_values = weight1_data.as_slice::<f32>().unwrap();
    let weight2_values = weight2_data.as_slice::<f32>().unwrap();
    
    for (w1, w2) in weight1_values.iter().zip(weight2_values.iter()) {
        assert!((w1 - w2).abs() < 1e-8,
                "LayerNorm weights should be identical: {} vs {}", w1, w2);
    }
    
    // LayerNorm biases should also be identical (typically initialized to 0.0)
    if let (Some(bias1), Some(bias2)) = (layer_norm1.bias.as_ref(), layer_norm2.bias.as_ref()) {
        let bias1_data = bias1.val().to_data();
        let bias2_data = bias2.val().to_data();
        
        let bias1_values = bias1_data.as_slice::<f32>().unwrap();
        let bias2_values = bias2_data.as_slice::<f32>().unwrap();
        
        for (b1, b2) in bias1_values.iter().zip(bias2_values.iter()) {
            assert!((b1 - b2).abs() < 1e-8,
                    "LayerNorm biases should be identical: {} vs {}", b1, b2);
        }
    }
}

#[test]
fn test_deterministic_embedding_reproducibility() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test embedding layer reproducibility
    let seed = 9999u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    let vocab_size = 100;
    let embed_dim = 16;
    
    let embedding1 = rng_ctx1.create_deterministic_embedding(vocab_size, embed_dim, 0);
    let embedding2 = rng_ctx2.create_deterministic_embedding(vocab_size, embed_dim, 0);
    
    // Embedding weights should be identical
    let weight1_data = embedding1.weight.val().to_data();
    let weight2_data = embedding2.weight.val().to_data();
    
    let weight1_values = weight1_data.as_slice::<f32>().unwrap();
    let weight2_values = weight2_data.as_slice::<f32>().unwrap();
    
    assert_eq!(weight1_values.len(), weight2_values.len(), 
               "Embedding weight tensors should have same size");
    
    for (w1, w2) in weight1_values.iter().zip(weight2_values.iter()) {
        assert!((w1 - w2).abs() < 1e-8,
                "Embedding weights should be identical: {} vs {}", w1, w2);
    }
}

#[test]
fn test_seed_offset_produces_different_weights() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that seed offsets produce different but reproducible weights
    let base_seed = 42u64;
    let rng_ctx = DeterministicRngContext::<TestBackend>::new(base_seed, device.clone());
    
    // Create linear layers with different seed offsets
    let linear_offset_0 = rng_ctx.create_deterministic_linear(2, 2, true, 0);
    let linear_offset_100 = rng_ctx.create_deterministic_linear(2, 2, true, 100);
    let linear_offset_200 = rng_ctx.create_deterministic_linear(2, 2, true, 200);
    
    let weights_0_data = linear_offset_0.weight.val().to_data();
    let weights_100_data = linear_offset_100.weight.val().to_data();
    let weights_200_data = linear_offset_200.weight.val().to_data();
    
    let weights_0 = weights_0_data.as_slice::<f32>().unwrap();
    let weights_100 = weights_100_data.as_slice::<f32>().unwrap();
    let weights_200 = weights_200_data.as_slice::<f32>().unwrap();
    
    // Different offsets should produce different weights
    let mut found_diff_0_100 = false;
    let mut found_diff_0_200 = false;
    let mut found_diff_100_200 = false;
    
    for ((w0, w100), w200) in weights_0.iter().zip(weights_100.iter()).zip(weights_200.iter()) {
        if (w0 - w100).abs() > 1e-6 { found_diff_0_100 = true; }
        if (w0 - w200).abs() > 1e-6 { found_diff_0_200 = true; }
        if (w100 - w200).abs() > 1e-6 { found_diff_100_200 = true; }
    }
    
    assert!(found_diff_0_100, "Offset 0 and 100 should produce different weights");
    assert!(found_diff_0_200, "Offset 0 and 200 should produce different weights");
    assert!(found_diff_100_200, "Offset 100 and 200 should produce different weights");
}

#[test]
fn test_reproducible_random_tensor_generation() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that we can generate reproducible random tensors
    let seed = 777u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    // Generate random tensors using the same seed
    let tensor1 = rng_ctx1.generate_normal_tensor([4, 4], 0.0, 1.0, Some(0));
    let tensor2 = rng_ctx2.generate_normal_tensor([4, 4], 0.0, 1.0, Some(0));
    
    let data1 = tensor1.to_data();
    let data2 = tensor2.to_data();
    
    let values1 = data1.as_slice::<f32>().unwrap();
    let values2 = data2.as_slice::<f32>().unwrap();
    
    // Values should be identical
    for (v1, v2) in values1.iter().zip(values2.iter()) {
        assert!((v1 - v2).abs() < 1e-8,
                "Random tensors should be identical with same seed: {} vs {}", v1, v2);
    }
    
    // Verify values are actually random (not all zeros or ones)
    let mut has_variety = false;
    let first_value = values1[0];
    for &value in values1.iter().skip(1) {
        if (value - first_value).abs() > 1e-6 {
            has_variety = true;
            break;
        }
    }
    assert!(has_variety, "Random tensor should have variety in values");
}

#[test]
fn test_deterministic_operations_chain() {
    let device = <TestBackend as Backend>::Device::default();
    
    // Test that chaining deterministic operations maintains reproducibility
    let seed = 555u64;
    let rng_ctx1 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    let rng_ctx2 = DeterministicRngContext::<TestBackend>::new(seed, device.clone());
    
    // Create a chain of operations with both contexts
    let linear1_a = rng_ctx1.create_deterministic_linear(3, 4, true, 0);
    let linear2_a = rng_ctx1.create_deterministic_linear(4, 2, true, 100);
    let layer_norm_a = rng_ctx1.create_deterministic_layer_norm(2, 1e-5);
    
    let linear1_b = rng_ctx2.create_deterministic_linear(3, 4, true, 0);
    let linear2_b = rng_ctx2.create_deterministic_linear(4, 2, true, 100);
    let layer_norm_b = rng_ctx2.create_deterministic_layer_norm(2, 1e-5);
    
    // Compare weights from first linear layer
    let weights1_a_data = linear1_a.weight.val().to_data();
    let weights1_b_data = linear1_b.weight.val().to_data();
    
    let weights1_a = weights1_a_data.as_slice::<f32>().unwrap();
    let weights1_b = weights1_b_data.as_slice::<f32>().unwrap();
    
    for (wa, wb) in weights1_a.iter().zip(weights1_b.iter()) {
        assert!((wa - wb).abs() < 1e-8,
                "Chained operations should produce identical results: {} vs {}", wa, wb);
    }
    
    // Compare weights from second linear layer
    let weights2_a_data = linear2_a.weight.val().to_data();
    let weights2_b_data = linear2_b.weight.val().to_data();
    
    let weights2_a = weights2_a_data.as_slice::<f32>().unwrap();
    let weights2_b = weights2_b_data.as_slice::<f32>().unwrap();
    
    for (wa, wb) in weights2_a.iter().zip(weights2_b.iter()) {
        assert!((wa - wb).abs() < 1e-8,
                "Second layer in chain should produce identical results: {} vs {}", wa, wb);
    }
}
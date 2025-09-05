use std::collections::HashMap;
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;

// Re-export fixture loading utilities
mod fixture_loader;
use fixture_loader::{FixtureLoader, FixtureCase, FixtureConfig};

// Import the MultiHeadAttention implementation
use tab_pfn_rs::tabpfn::architectures::base::attention::full_attention::MultiHeadAttention;
use tab_pfn_rs::tabpfn::architectures::base::attention::Attention;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;

type TestBackend = NdArray<f32>;

/// Test tolerance for floating point comparisons
const TOLERANCE: f64 = 1e-6;

/// Helper function to compare two tensors with tolerance
fn assert_tensors_close<const D: usize>(
    actual: &Tensor<TestBackend, D>,
    expected: &Tensor<TestBackend, D>, 
    tolerance: f64,
    description: &str,
) {
    let actual_shape = actual.shape().dims;
    let expected_shape = expected.shape().dims;
    
    assert_eq!(actual_shape, expected_shape, 
        "{}: Shape mismatch - actual {:?} vs expected {:?}", description, actual_shape, expected_shape);
    
    // Convert to data for comparison
    let actual_data: Vec<f32> = actual.clone().into_data().to_vec().unwrap();
    let expected_data: Vec<f32> = expected.clone().into_data().to_vec().unwrap();
    
    assert_eq!(actual_data.len(), expected_data.len(), 
        "{}: Data length mismatch", description);
    
    let mut max_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    
    for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = (a - e).abs() as f64;
        let rel_diff = if e.abs() > 1e-10 { diff / (e.abs() as f64) } else { diff };
        
        max_diff = max_diff.max(diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        
        if diff > tolerance && rel_diff > tolerance {
            panic!("{}: Value mismatch at index {}: actual={}, expected={}, abs_diff={}, rel_diff={}", 
                description, i, a, e, diff, rel_diff);
        }
    }
    
    println!("✓ {}: max_abs_diff={:.2e}, max_rel_diff={:.2e}", description, max_diff, max_rel_diff);
}

/// Helper function to convert fixture data to tensor
fn fixture_data_to_tensor<const D: usize>(
    shape: Vec<usize>, 
    data: Vec<f32>,
    device: &<TestBackend as burn::prelude::Backend>::Device
) -> Tensor<TestBackend, D> {
    assert_eq!(shape.len(), D, "Shape dimension mismatch");
    let tensor_data = TensorData::new(data, shape.clone());
    Tensor::<TestBackend, D>::from_data(tensor_data, device)
}

/// Create ModelConfig from fixture config
fn create_model_config(config: &FixtureConfig) -> ModelConfig {
    let mut model_config = ModelConfig::default();
    model_config.emsize = config.emsize as i32;
    model_config.nhead = config.nhead as i32;
    model_config.dropout = config.dropout_p.unwrap_or(0.0);
    model_config.max_num_classes = 100; // Reasonable default for tests
    model_config.num_buckets = 100;     // Reasonable default for tests
    model_config
}

/// Test end-to-end streaming scenario
fn test_streaming_scenario(case: &FixtureCase, fixture_data: HashMap<String, (Vec<usize>, Vec<f32>)>) {
    println!("\n🧪 Testing streaming scenario: {}", case.case_id);
    println!("   Config: batch={}, seq_q={}, seq_kv={}, emsize={}, nhead={}, d_k={}, d_v={}", 
        case.config.batch_size, case.config.seq_len_q, case.config.seq_len_kv,
        case.config.emsize, case.config.nhead, case.config.d_k, case.config.d_v);
    
    let device: <TestBackend as Backend>::Device = Default::default();
    let model_config = create_model_config(&case.config);
    let rng_context = DeterministicRngContext::<TestBackend>::new(case.seed, device.clone());
    
    // Create MultiHeadAttention instance with deterministic mode for testing
    let mut attention = MultiHeadAttention::<TestBackend>::new(
        case.config.d_k,
        case.config.d_v,
        &model_config,
        case.config.share_kv_across_n_heads,
        case.config.dropout_p,
        None, // softmax_scale
        false, // initialize_output_to_zero
        None, // precomputed_k
        None, // precomputed_v  
        None, // precomputed_kv
        &rng_context, // rng_ctx
        case.seed, // init_seed_offset
        true, // inference_mode
    );
    
    // Load weights from fixture
    println!("📥 Loading fixture weights...");
    let mut weights: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    
    // Load all weights from fixture
    for (key, value) in &fixture_data {
        if key.starts_with("weight_") {
            weights.insert(key.clone(), value.clone());
        }
    }
    
    attention.load_weights_from_numpy(weights, &device)
        .expect("Failed to load weights from fixture");
    
    println!("📥 Fixture weights loaded.");
    
    // Clear any existing caches
    attention.empty_kv_cache();
    
    // Phase 1: Load chunk inputs
    let input_x_chunk1 = {
        let (shape, data) = fixture_data.get("inputs_x_chunk1")
            .or_else(|| fixture_data.get("input_x_chunk1"))
            .expect("Missing chunk1 input in fixture");
        println!("📥 Input x_chunk1: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    let input_x_chunk2 = {
        let (shape, data) = fixture_data.get("inputs_x_chunk2")
            .or_else(|| fixture_data.get("input_x_chunk2"))
            .expect("Missing chunk2 input in fixture");
        println!("📥 Input x_chunk2: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    // Phase 2: Process first chunk with caching
    println!("🚀 Processing chunk 1 with caching...");
    let output_chunk1 = attention.forward(
        input_x_chunk1.clone(),
        None, // x_kv (self-attention)
        true, // cache_kv - populate the cache
        false, // use_cached_kv
        false, // reuse_first_head_kv
        false, // only_cache_first_head_kv
        None, // save_peak_mem_factor
        false, // add_input
        false, // allow_inplace
        false, // train
        None, // attention_mask
    );
    
    // Verify cache is populated
    assert!(attention.has_cached_kv(), "Cache should be populated after chunk 1");
    println!("✓ Cache populated after chunk 1");
    
    // Phase 3: Process second chunk using cached KV
    println!("🚀 Processing chunk 2 with cached KV...");
    let output_chunk2 = attention.forward(
        input_x_chunk2.clone(),
        None, // x_kv (self-attention)
        false, // cache_kv
        true, // use_cached_kv - use the populated cache
        false, // reuse_first_head_kv
        false, // only_cache_first_head_kv
        None, // save_peak_mem_factor
        false, // add_input
        false, // allow_inplace
        false, // train
        None, // attention_mask
    );
    
    println!("✓ Chunk processing completed");
    
    // Phase 4: Compare with expected outputs
    let expected_output_chunk1 = {
        let (shape, data) = fixture_data.get("outputs_output_chunk1")
            .or_else(|| fixture_data.get("output_output_chunk1"))
            .expect("Missing expected output_chunk1 in fixture");
        println!("📤 Expected output_chunk1: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    let expected_output_chunk2 = {
        let (shape, data) = fixture_data.get("outputs_output_chunk2")
            .or_else(|| fixture_data.get("output_output_chunk2"))
            .expect("Missing expected output_chunk2 in fixture");
        println!("📤 Expected output_chunk2: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    // Phase 5: Validate chunk outputs
    assert_tensors_close(&output_chunk1, &expected_output_chunk1, TOLERANCE, "Chunk 1 output");
    assert_tensors_close(&output_chunk2, &expected_output_chunk2, TOLERANCE, "Chunk 2 output");
    
    // Phase 6: Full sequence reference validation (if available)
    if let Some((shape, data)) = fixture_data.get("outputs_output_full_reference")
        .or_else(|| fixture_data.get("output_output_full_reference")) {
        
        println!("🔍 Validating against full sequence reference...");
        
        // Process full sequence for comparison
        attention.empty_kv_cache(); // Clear cache
        let full_input = {
            let (shape, data) = fixture_data.get("input_x")
                .expect("Missing full input_x for reference comparison");
            fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
        };
        
        let full_output = attention.forward(
            full_input,
            None, // x_kv (self-attention)
            false, // cache_kv
            false, // use_cached_kv
            false, // reuse_first_head_kv
            false, // only_cache_first_head_kv
            None, // save_peak_mem_factor
            false, // add_input
            false, // allow_inplace
            false, // train
            None, // attention_mask
        );
        
        let expected_full_output = fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device);
        
        assert_tensors_close(&full_output, &expected_full_output, TOLERANCE, "Full sequence reference");
        
        // Concatenate chunk outputs and compare with full reference
        let concatenated_chunks = Tensor::cat(vec![output_chunk1, output_chunk2], 1); // Concatenate on seq dimension
        
        // Note: For some streaming scenarios, the concatenated chunks may not exactly match
        // the full sequence due to different attention patterns. This is expected behavior.
        println!("📊 Chunk concatenation shape: {:?}", concatenated_chunks.shape().dims);
        println!("📊 Full reference shape: {:?}", expected_full_output.shape().dims);
        
        if concatenated_chunks.shape().dims == expected_full_output.shape().dims {
            println!("🔍 Comparing concatenated chunks with full reference...");
            // Allow slightly higher tolerance for streaming vs. full sequence comparison
            let streaming_tolerance = TOLERANCE * 10.0;
            assert_tensors_close(&concatenated_chunks, &expected_full_output, streaming_tolerance, "Concatenated chunks vs full sequence");
        } else {
            println!("ℹ️  Shape mismatch expected for this streaming scenario - skipping concatenation comparison");
        }
    }
    
    println!("✅ Streaming scenario {} PASSED", case.case_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_streaming_end_to_end() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("streaming").expect("Missing streaming test case");
        let fixture_data = loader.load_simple_test_fixture("streaming.test")
            .expect("Failed to load streaming fixture data");
        
        test_streaming_scenario(case, fixture_data);
    }
    
    #[test]
    fn test_streaming_state_consistency() {
        println!("\n🧪 Testing streaming with state save/load consistency");
        
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("streaming").expect("Missing streaming test case");
        let fixture_data = loader.load_simple_test_fixture("streaming.test")
            .expect("Failed to load streaming fixture data");
        
        let device: <TestBackend as Backend>::Device = Default::default();
        let model_config = create_model_config(&case.config);
        
        // Create first attention instance
        let rng_context = DeterministicRngContext::<TestBackend>::new(case.seed, device.clone());
        let mut attention1 = MultiHeadAttention::<TestBackend>::new(
            case.config.d_k,
            case.config.d_v,
            &model_config,
            case.config.share_kv_across_n_heads,
            case.config.dropout_p,
            None, // softmax_scale
            false, // initialize_output_to_zero
            None, // precomputed_k
            None, // precomputed_v
            None, // precomputed_kv
            &rng_context, // rng_ctx
            case.seed, // init_seed_offset
            true, // inference_mode
        );
        
        // Load weights
        let mut weights: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (key, value) in &fixture_data {
            if key.starts_with("weight_") {
                weights.insert(key.clone(), value.clone());
            }
        }
        attention1.load_weights_from_numpy(weights.clone(), &device).unwrap();
        
        // Process first chunk
        let input_x_chunk1 = fixture_data.get("input_x_chunk1")
            .or_else(|| fixture_data.get("inputs_x_chunk1"))
            .map(|(shape, data)| fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device))
            .expect("Missing chunk1 input");
        
        let _output1 = attention1.forward(input_x_chunk1, None, true, false, false, false, None, false, false, false, None);
        
        // Save state after first chunk
        println!("💾 Saving state after chunk 1...");
        let state = attention1.save_state();
        
        // Create second attention instance and load state
        let rng_context2 = DeterministicRngContext::<TestBackend>::new(case.seed, device.clone());
        let mut attention2 = MultiHeadAttention::<TestBackend>::new(
            case.config.d_k,
            case.config.d_v,
            &model_config,
            case.config.share_kv_across_n_heads,
            case.config.dropout_p,
            None, // softmax_scale
            false, // initialize_output_to_zero
            None, // precomputed_k
            None, // precomputed_v
            None, // precomputed_kv
            &rng_context2, // rng_ctx
            case.seed, // init_seed_offset
            true, // inference_mode
        );
        
        println!("📥 Loading state into new instance...");
        attention2.load_state(state, &device).expect("Failed to load state");
        
        // Verify cache is present in loaded instance
        assert!(attention2.has_cached_kv(), "Cache should be present after state load");
        
        // Process second chunk with loaded state
        let input_x_chunk2 = fixture_data.get("input_x_chunk2")
            .or_else(|| fixture_data.get("inputs_x_chunk2"))
            .map(|(shape, data)| fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device))
            .expect("Missing chunk2 input");
            
        let output2_from_loaded_state = attention2.forward(input_x_chunk2.clone(), None, false, true, false, false, None, false, false, false, None);
        
        // Compare with original instance processing chunk 2
        let output2_from_original = attention1.forward(input_x_chunk2, None, false, true, false, false, None, false, false, false, None);
        
        assert_tensors_close(&output2_from_loaded_state, &output2_from_original, TOLERANCE, "Chunk 2 output: loaded state vs original");
        
        println!("✅ Streaming state consistency test PASSED");
    }
    
    #[test] 
    fn test_state_roundtrip_validation() {
        println!("\n🧪 Testing state save/load roundtrip validation");
        
        let device: <TestBackend as Backend>::Device = Default::default();
        let model_config = ModelConfig::default();
        
        // Create attention instance with deterministic weights
        let rng_context3 = DeterministicRngContext::<TestBackend>::new(42, device.clone());
        let mut attention = MultiHeadAttention::<TestBackend>::new(
            16, // d_k
            16, // d_v
            &model_config,
            1, // share_kv_across_n_heads
            None, // dropout_p
            None, // softmax_scale
            false, // initialize_output_to_zero
            None, // precomputed_k
            None, // precomputed_v
            None, // precomputed_kv
            &rng_context3, // rng_ctx
            42, // init_seed_offset
            true, // inference_mode
        );
        
        // Process some dummy input to populate cache
        let dummy_input = Tensor::<TestBackend, 3>::random([1, 4, 32], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let original_output = attention.forward(dummy_input.clone(), None, true, false, false, false, None, false, false, false, None);
        
        // Save state
        println!("💾 Saving original state...");
        let original_state = attention.save_state();
        
        // Save to file and load back
        let temp_file = "/tmp/test_attention_state.bin";
        attention.save_to_file(temp_file).expect("Failed to save to file");
        
        // Create new instance and load from file
        let rng_context4 = DeterministicRngContext::<TestBackend>::new(99, device.clone());
        let mut attention_loaded = MultiHeadAttention::<TestBackend>::new(
            16, // d_k
            16, // d_v
            &model_config,
            1, // share_kv_across_n_heads
            None, // dropout_p
            None, // softmax_scale
            false, // initialize_output_to_zero
            None, // precomputed_k
            None, // precomputed_v
            None, // precomputed_kv
            &rng_context4, // rng_ctx
            99, // init_seed_offset - Different seed to ensure loading works
            true, // inference_mode
        );
        
        println!("📥 Loading state from file...");
        attention_loaded.load_from_file(temp_file, &device).expect("Failed to load from file");
        
        // Process same input and compare outputs
        let loaded_output = attention_loaded.forward(dummy_input.clone(), None, false, true, false, false, None, false, false, false, None);
        
        assert_tensors_close(&loaded_output, &original_output, TOLERANCE, "Roundtrip state validation");
        
        // Clean up
        std::fs::remove_file(temp_file).ok();
        
        println!("✅ State roundtrip validation PASSED");
    }
}
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

type TestBackend = NdArray<f32>;

/// Test tolerance for floating point comparisons (≤1e-6 as required)
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

/// Test a single fixture case
fn test_fixture_case(case: &FixtureCase, fixture_data: HashMap<String, (Vec<usize>, Vec<f32>)>) {
    println!("\n🧪 Testing case: {}", case.case_id);
    println!("   Config: batch={}, seq_q={}, seq_kv={}, emsize={}, nhead={}, d_k={}, d_v={}", 
        case.config.batch_size, case.config.seq_len_q, case.config.seq_len_kv,
        case.config.emsize, case.config.nhead, case.config.d_k, case.config.d_v);
    
    let device = Default::default();
    let model_config = create_model_config(&case.config);
    
    // Create MultiHeadAttention instance with deterministic mode for testing
    let mut attention = MultiHeadAttention::<TestBackend>::new(
        case.config.d_k,
        case.config.d_v,
        &device,
        &model_config,
        case.config.share_kv_across_n_heads,
        case.config.dropout_p,
        None, // softmax_scale
        false, // initialize_output_to_zero
        None, // precomputed_k
        None, // precomputed_v  
        None, // precomputed_kv
        true, // deterministic_init
        Some(case.seed), // init_seed
        true, // inference_mode
    );
    
    // Load input tensors from fixture
    let input_x = {
        let (shape, data) = fixture_data.get("input_x")
            .expect("Missing input_x in fixture");
        println!("📥 Input x: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    // For use_cached_kv scenario, there might be a separate query input 
    let actual_query_input = if case.config.cache_scenario == "use_cached" {
        if let Some((shape, data)) = fixture_data.get("input_x_new_query") {
            println!("📥 Input x_new_query: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
        } else {
            input_x.clone()
        }
    } else {
        input_x.clone()
    };
    
    let input_x_kv = if case.config.use_self_attention {
        println!("📥 Using self-attention (no x_kv)");
        None
    } else {
        fixture_data.get("input_x_kv").map(|(shape, data)| {
            println!("📥 Input x_kv: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
        })
    };
    
    // Load and set weight tensors from fixture
    println!("📥 Loading fixture weights...");
    
    if let Some((shape, data)) = fixture_data.get("weight_w_q") {
        if shape.len() == 4 {
            println!("   Loading w_q: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_q_tensor = fixture_data_to_tensor::<4>(shape.clone(), data.clone(), &device);
            attention.set_w_q(Some(burn::module::Param::from_tensor(w_q_tensor)));
        }
    } else {
        println!("   No w_q found in fixture");
    }
    
    if let Some((shape, data)) = fixture_data.get("weight_w_k") {
        if shape.len() == 3 {
            println!("   Loading w_k: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_k_tensor = fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device);
            attention.set_w_k(Some(burn::module::Param::from_tensor(w_k_tensor)));
        }
    } else {
        println!("   No w_k found in fixture");
    }
    
    if let Some((shape, data)) = fixture_data.get("weight_w_v") {
        if shape.len() == 3 {
            println!("   Loading w_v: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_v_tensor = fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device);
            attention.set_w_v(Some(burn::module::Param::from_tensor(w_v_tensor)));
        }
    } else {
        println!("   No w_v found in fixture");
    }
    
    if let Some((shape, data)) = fixture_data.get("weight_w_kv") {
        if shape.len() == 4 {
            println!("   Loading w_kv: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_kv_tensor = fixture_data_to_tensor::<4>(shape.clone(), data.clone(), &device);
            attention.set_w_kv(Some(burn::module::Param::from_tensor(w_kv_tensor)));
        }
    } else {
        println!("   No w_kv found in fixture");
    }
    
    if let Some((shape, data)) = fixture_data.get("weight_w_qkv") {
        if shape.len() == 4 {
            println!("   Loading w_qkv: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_qkv_tensor = fixture_data_to_tensor::<4>(shape.clone(), data.clone(), &device);
            attention.set_w_qkv(Some(burn::module::Param::from_tensor(w_qkv_tensor)));
        }
    } else {
        println!("   No w_qkv found in fixture");
    }
    
    if let Some((shape, data)) = fixture_data.get("weight_w_out") {
        if shape.len() == 3 {
            println!("   Loading w_out: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let w_out_tensor = fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device);
            attention.set_w_out(burn::module::Param::from_tensor(w_out_tensor));
        }
    } else {
        println!("   No w_out found in fixture");
    }
    
    println!("📥 Fixture weights loaded.");
    
    // Clear any existing caches for deterministic test isolation
    attention.empty_kv_cache();
    
    // Handle caching scenarios (mutually exclusive)
    let cache_kv = case.config.cache_scenario == "cache_kv";
    let use_cached_kv = case.config.cache_scenario == "use_cached";
    
    // For use_cached_kv scenario, we need to pre-populate the cache with fixture data
    if use_cached_kv {
        println!("📥 Loading pre-computed cache for use_cached_kv scenario...");
        
        if let Some((shape, data)) = fixture_data.get("cache_kv_cache_used") {
            println!("   Loading kv_cache: shape={:?}, sample_values={:?}", shape, &data[..5.min(data.len())]);
            let kv_cache_tensor = fixture_data_to_tensor::<5>(shape.clone(), data.clone(), &device);
            
            // For use_cached_kv, we need a different approach since set_parameters doesn't allow both weights and cache
            // Instead, we'll simulate the scenario by first running with cache_kv=true to populate cache,
            // then using that populated cache for the actual test
            println!("   Using alternative approach: simulating cache population first...");
            
            // Run a dummy forward pass with cache_kv=true to populate the cache
            // We need to use the same input that was used to generate the cached values in Python
            let dummy_input = input_x.clone(); 
            
            println!("   Running cache population pass...");
            let _dummy_output = attention.forward(
                dummy_input,
                None, // x_kv (self-attention)
                true, // cache_kv - populate the cache
                false, // use_cached_kv
                false, // reuse_first_head_kv
                false, // only_cache_first_head_kv
                None, // save_peak_mem_factor
                false, // add_input
                false, // allow_inplace
            );
            
            println!("   Cache population completed.");
            
            // Verify cache is actually set
            if attention.has_cached_kv() {
                println!("   ✓ Cache verification: has_cached_kv() = true");
                if let Some(cache) = attention.get_kv_cache() {
                    println!("   ✓ KV cache shape: {:?}", cache.shape().dims);
                }
            } else {
                println!("   ❌ Cache verification: has_cached_kv() = false");
            }
        } else if let Some((k_shape, k_data)) = fixture_data.get("cache_k_cache_used") {
            if let Some((v_shape, v_data)) = fixture_data.get("cache_v_cache_used") {
                println!("   Loading k_cache: shape={:?}, v_cache: shape={:?}", k_shape, v_shape);
                let k_cache_tensor = fixture_data_to_tensor::<4>(k_shape.clone(), k_data.clone(), &device);
                let v_cache_tensor = fixture_data_to_tensor::<4>(v_shape.clone(), v_data.clone(), &device);
                
                let current_w_out = attention.w_out();
                match attention.set_parameters(
                    current_w_out,
                    attention.w_q(),
                    attention.w_k(),
                    attention.w_v(), 
                    attention.w_kv(),
                    attention.w_qkv(),
                    Some(k_cache_tensor), // precomputed_k
                    Some(v_cache_tensor), // precomputed_v
                    None, // precomputed_kv
                ) {
                    Ok(_) => println!("   ✓ Successfully set k/v cache via set_parameters"),
                    Err(e) => println!("   ❌ Failed to set k/v cache: {}", e),
                }
                
                // Verify cache is actually set
                if attention.has_cached_kv() {
                    println!("   ✓ Cache verification: has_cached_kv() = true");
                    if let Some(k_cache) = attention.get_k_cache() {
                        println!("   ✓ K cache shape: {:?}", k_cache.shape().dims);
                    }
                    if let Some(v_cache) = attention.get_v_cache() {
                        println!("   ✓ V cache shape: {:?}", v_cache.shape().dims);
                    }
                } else {
                    println!("   ❌ Cache verification: has_cached_kv() = false");
                }
            }
        }
        
        println!("📥 Cache pre-loaded for use_cached_kv scenario.");
    }
    
    // Run forward pass
    println!("🚀 Running forward pass...");
    let output = attention.forward(
        actual_query_input, // Use the appropriate query input (might be different for use_cached scenarios)
        input_x_kv,
        cache_kv,
        use_cached_kv,
        false, // reuse_first_head_kv
        false, // only_cache_first_head_kv
        None,  // save_peak_mem_factor
        false, // add_input
        false, // allow_inplace  
    );
    
    // Compare with expected output (handle different output key names for different scenarios)
    let output_key = match case.config.cache_scenario.as_str() {
        "cache_kv" => "output_output_with_cache",
        "use_cached" => "output_output_with_cached_kv", 
        "streaming" => "output_output_full_reference",
        _ => "output_output"
    };
    
    let expected_output = {
        let (shape, data) = fixture_data.get(output_key)
            .or_else(|| fixture_data.get("output_output"))
            .expect(&format!("Missing {} or output_output in fixture", output_key));
        println!("📤 Expected output ({}): shape={:?}, sample_values={:?}", output_key, shape, &data[..5.min(data.len())]);
        fixture_data_to_tensor::<3>(shape.clone(), data.clone(), &device)
    };
    
    // Print actual output for debugging
    let actual_data: Vec<f32> = output.clone().into_data().to_vec().unwrap();
    println!("📤 Actual output: shape={:?}, sample_values={:?}", output.shape().dims, &actual_data[..5.min(actual_data.len())]);
    
    // Simple mathematical check for constant values
    if fixture_data.get("weight_w_qkv").is_some() && fixture_data.get("weight_w_out").is_some() {
        println!("🔍 Performing manual computation check for constant values...");
        
        // Expected simple computation with all-constant values:
        // QKV projection: 0.1 * 0.01 = 0.001 (per element)
        // After attention mechanism with scaling
        // Output projection: attention_result * 0.02
        
        // For dummy data, all elements should follow the same pattern
        let input_val = 0.1_f32;
        let qkv_weight_val = 0.01_f32;
        let out_weight_val = 0.02_f32;
        let expected_val = 0.05_f32;
        let actual_val = actual_data[0];
        
        println!("   Input value: {}", input_val);
        println!("   QKV weight value: {}", qkv_weight_val);
        println!("   Out weight value: {}", out_weight_val);
        println!("   Expected output: {}", expected_val);
        println!("   Actual output: {}", actual_val);
        println!("   Ratio (expected/actual): {:.3}", expected_val / actual_val);
        
        // Simple linear approximation (ignoring softmax complexity):
        // If computation was: input * qkv_weight * out_weight * some_factor
        // Then: 0.1 * 0.01 * 0.02 * factor = expected
        // 0.00002 * factor = 0.05
        // factor = 2500
        let naive_computation = input_val * qkv_weight_val * out_weight_val;
        let implied_factor = expected_val / naive_computation;
        println!("   Naive computation (input*qkv*out): {}", naive_computation);
        println!("   Implied scaling factor: {:.1}", implied_factor);
    }
    
    assert_tensors_close(&output, &expected_output, TOLERANCE, "Forward output");
    
    // Compare cached values if applicable
    if cache_kv {
        if let Some((shape, data)) = fixture_data.get("cache_k_cache") {
            if let Some(k_cache) = attention.get_k_cache() {
                let expected_k_cache = fixture_data_to_tensor::<4>(shape.clone(), data.clone(), &device);
                assert_tensors_close(&k_cache, &expected_k_cache, TOLERANCE, "K cache");
            }
        }
        
        if let Some((shape, data)) = fixture_data.get("cache_v_cache") {
            if let Some(v_cache) = attention.get_v_cache() {
                let expected_v_cache = fixture_data_to_tensor::<4>(shape.clone(), data.clone(), &device);
                assert_tensors_close(&v_cache, &expected_v_cache, TOLERANCE, "V cache");
            }
        }
        
        if let Some((shape, data)) = fixture_data.get("cache_kv_cache") {
            if let Some(kv_cache) = attention.get_kv_cache() {
                let expected_kv_cache = fixture_data_to_tensor::<5>(shape.clone(), data.clone(), &device);
                assert_tensors_close(&kv_cache, &expected_kv_cache, TOLERANCE, "KV cache");
            }
        }
    }
    
    println!("✅ Case {} PASSED", case.case_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_parity_basic_self_attention() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("basic_self_attn_small").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("basic_self_attn_small.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_cross_attention() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("cross_attention").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("cross_attention.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_different_dk_dv() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("different_dk_dv").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("different_dk_dv.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]  
    fn test_attention_parity_kv_sharing() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("kv_sharing").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("kv_sharing.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_with_caching() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("with_caching").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("with_caching.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_use_cached_kv() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("use_cached_kv").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("use_cached_kv.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_streaming() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("streaming").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("streaming.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    #[test]
    fn test_attention_parity_large_batch_dropout() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let case = loader.get_case_by_id("large_batch_dropout").expect("Missing test case");
        let fixture_data = loader.load_simple_test_fixture("large_batch_dropout.test")
            .expect("Failed to load fixture data");
        
        test_fixture_case(case, fixture_data);
    }
    
    /// Comprehensive test that runs all fixture cases
    #[test]
    fn test_all_attention_parity_cases() {
        let loader = FixtureLoader::new("fixtures").expect("Failed to load fixture loader");
        let cases = loader.list_cases();
        let total_cases = cases.len();
        
        println!("🚀 Running comprehensive parity tests for {} cases", total_cases);
        
        let mut passed = 0;
        let mut failed = 0;
        
        for case_id in &cases {
            let case = loader.get_case_by_id(case_id).expect("Missing test case");
            let fixture_filename = format!("{}.test", case_id);
            
            match loader.load_simple_test_fixture(&fixture_filename) {
                Ok(fixture_data) => {
                    match std::panic::catch_unwind(|| {
                        test_fixture_case(case, fixture_data);
                    }) {
                        Ok(_) => {
                            passed += 1;
                        },
                        Err(e) => {
                            failed += 1;
                            println!("❌ Case {} FAILED: {:?}", case_id, e);
                        }
                    }
                },
                Err(e) => {
                    failed += 1;
                    println!("❌ Case {} FAILED to load fixture: {}", case_id, e);
                }
            }
        }
        
        println!("\n📊 Parity Test Summary: {} passed, {} failed", passed, failed);
        
        if failed > 0 {
            panic!("Some parity tests failed! {} out of {} cases failed", failed, total_cases);
        } else {
            println!("🎉 ALL PARITY TESTS PASSED! Rust implementation matches Python reference within tolerance ≤{:.0e}", TOLERANCE);
        }
    }
}
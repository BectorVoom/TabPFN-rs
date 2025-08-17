use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;

// Import the MultiHeadAttention implementation and test utilities
use tab_pfn_rs::tabpfn::architectures::base::attention::full_attention::MultiHeadAttention;
use tab_pfn_rs::tabpfn::architectures::base::attention::Attention;
use tab_pfn_rs::tabpfn::architectures::base::config::ModelConfig;

type BenchBackend = NdArray<f32>;

/// Helper function to create deterministic attention instances
fn create_attention(
    d_k: usize, 
    d_v: usize, 
    emsize: usize,
    nhead: usize,
    device: &<BenchBackend as Backend>::Device,
) -> MultiHeadAttention<BenchBackend> {
    let mut config = ModelConfig::default();
    config.emsize = emsize as i32;
    config.nhead = nhead as i32;
    config.dropout = 0.0; // No dropout for benchmarking
    
    MultiHeadAttention::<BenchBackend>::new(
        d_k,
        d_v,
        device,
        &config,
        1, // share_kv_across_n_heads
        None, // dropout_p
        None, // softmax_scale
        false, // initialize_output_to_zero
        None, // precomputed_k
        None, // precomputed_v
        None, // precomputed_kv
        true, // deterministic_init
        Some(42), // init_seed for consistency
        true, // inference_mode
    )
}

/// Helper function to create test input tensors
fn create_test_input(
    batch_size: usize,
    seq_len: usize,
    emsize: usize,
    device: &<BenchBackend as Backend>::Device,
) -> Tensor<BenchBackend, 3> {
    // Create deterministic input for consistent benchmarking
    let data: Vec<f32> = (0..batch_size * seq_len * emsize)
        .map(|i| (i as f32 * 0.01).sin()) // Deterministic pattern
        .collect();
    
    let tensor_data = TensorData::new(data, [batch_size, seq_len, emsize]);
    Tensor::from_data(tensor_data, device)
}

/// Benchmark different attention configurations
fn benchmark_attention_configurations(c: &mut Criterion) {
    let device = Default::default();
    
    let configs = vec![
        // (name, batch, seq_len, emsize, nhead, d_k, d_v)
        ("small", 1, 32, 64, 4, 16, 16),
        ("medium", 1, 128, 128, 8, 16, 16), 
        ("large", 1, 256, 256, 8, 32, 32),
        ("xlarge", 2, 512, 512, 16, 32, 32),
    ];
    
    let mut group = c.benchmark_group("attention_forward_pass");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    
    for (name, batch, seq_len, emsize, nhead, d_k, d_v) in configs {
        let input = create_test_input(batch, seq_len, emsize, &device);
        
        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &(d_k, d_v, emsize, nhead, input),
            |b, &(d_k, d_v, emsize, nhead, ref input)| {
                b.iter_with_setup(
                    || create_attention(d_k, d_v, emsize, nhead, &device),
                    |mut attention| {
                        black_box(attention.forward(
                            black_box(input.clone()),
                            None, // x_kv (self-attention)
                            false, // cache_kv
                            false, // use_cached_kv
                            false, // reuse_first_head_kv
                            false, // only_cache_first_head_kv
                            None, // save_peak_mem_factor
                            false, // add_input
                            false, // allow_inplace
                        ))
                    }
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark caching operations
fn benchmark_caching_operations(c: &mut Criterion) {
    let device = Default::default();
    let input = create_test_input(1, 64, 128, &device);
    
    let mut group = c.benchmark_group("attention_caching");
    group.measurement_time(Duration::from_secs(5));
    
    // Benchmark cache population
    group.bench_function("cache_kv_populate", |b| {
        b.iter_with_setup(
            || create_attention(16, 16, 128, 8, &device),
            |mut attention| {
                attention.empty_kv_cache(); // Reset for each iteration
                black_box(attention.forward(
                    black_box(input.clone()),
                    None, // x_kv
                    true, // cache_kv - populate cache
                    false, // use_cached_kv
                    false, false, None, false, false,
                ))
            }
        );
    });
    
    // Benchmark using cached KV (need to populate cache first)
    group.bench_function("use_cached_kv", |b| {
        b.iter_with_setup(
            || {
                let mut attention = create_attention(16, 16, 128, 8, &device);
                attention.empty_kv_cache();
                let _ = attention.forward(
                    input.clone(), None, true, false, false, false, None, false, false,
                );
                attention
            },
            |mut attention| {
                black_box(attention.forward(
                    black_box(input.clone()),
                    None, // x_kv
                    false, // cache_kv
                    true, // use_cached_kv - use populated cache
                    false, false, None, false, false,
                ))
            }
        );
    });
    
    group.finish();
}

/// Benchmark state management operations
fn benchmark_state_management(c: &mut Criterion) {
    let device = Default::default();
    let input = create_test_input(1, 32, 128, &device);
    
    let mut group = c.benchmark_group("state_management");
    group.measurement_time(Duration::from_secs(5));
    
    // Create attention with populated state once
    let mut attention_template = create_attention(16, 16, 128, 8, &device);
    let _ = attention_template.forward(
        input, None, true, false, false, false, None, false, false,
    );
    let template_state = attention_template.save_state();
    
    // Benchmark state save operation
    group.bench_function("save_state", |b| {
        b.iter_with_setup(
            || {
                let mut attention = create_attention(16, 16, 128, 8, &device);
                // Create new state instead of loading (to avoid validation conflicts)
                let input = create_test_input(1, 32, 128, &device);
                let _ = attention.forward(
                    input, None, true, false, false, false, None, false, false,
                );
                attention
            },
            |attention| {
                black_box(attention.save_state())
            }
        );
    });
    
    // Benchmark state load operation
    group.bench_function("load_state", |b| {
        b.iter_with_setup(
            || create_attention(16, 16, 128, 8, &device),
            |mut attention| {
                let state_clone = template_state.clone();
                black_box(attention.load_state(state_clone, &device).unwrap())
            }
        );
    });
    
    // Benchmark file save operation
    group.bench_function("save_to_file", |b| {
        b.iter_with_setup(
            || {
                let mut attention = create_attention(16, 16, 128, 8, &device);
                // Create new state instead of loading (to avoid validation conflicts)
                let input = create_test_input(1, 32, 128, &device);
                let _ = attention.forward(
                    input, None, true, false, false, false, None, false, false,
                );
                attention
            },
            |attention| {
                let temp_path = format!("/tmp/bench_state_{}.bin", std::process::id());
                black_box(attention.save_to_file(&temp_path).unwrap());
                std::fs::remove_file(&temp_path).ok(); // Cleanup
            }
        );
    });
    
    // Benchmark file load operation - create persistent file once
    let temp_path = "/tmp/bench_state_persistent.bin";
    attention_template.save_to_file(temp_path).unwrap();
    
    group.bench_function("load_from_file", |b| {
        b.iter_with_setup(
            || create_attention(16, 16, 128, 8, &device),
            |mut attention| {
                black_box(attention.load_from_file(temp_path, &device).unwrap())
            }
        );
    });
    
    // Cleanup
    std::fs::remove_file(temp_path).ok();
    group.finish();
}

/// Benchmark memory scaling patterns  
fn benchmark_memory_scaling(c: &mut Criterion) {
    let device = Default::default();
    
    let seq_lengths = vec![32, 64, 128, 256, 512];
    
    let mut group = c.benchmark_group("memory_scaling");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);
    
    for seq_len in seq_lengths {
        let input = create_test_input(1, seq_len, 128, &device);
        
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(seq_len, input),
            |b, &(seq_len, ref input)| {
                b.iter_with_setup(
                    || create_attention(16, 16, 128, 8, &device),
                    |mut attention| {
                        black_box(attention.forward(
                            black_box(input.clone()),
                            None, false, false, false, false, None, false, false,
                        ))
                    }
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark different head configurations
fn benchmark_head_scaling(c: &mut Criterion) {
    let device = Default::default();
    
    let head_configs = vec![
        // (nhead, d_k, d_v)
        (1, 64, 64),
        (2, 32, 32), 
        (4, 16, 16),
        (8, 16, 16),
        (16, 8, 8),
    ];
    
    let mut group = c.benchmark_group("head_scaling");
    group.measurement_time(Duration::from_secs(8));
    
    for (nhead, d_k, d_v) in head_configs {
        let emsize = nhead * d_k; // Keep total model size roughly constant
        let input = create_test_input(1, 128, emsize, &device);
        
        group.bench_with_input(
            BenchmarkId::new("heads", nhead),
            &(nhead, d_k, d_v, emsize, input),
            |b, &(nhead, d_k, d_v, emsize, ref input)| {
                b.iter_with_setup(
                    || create_attention(d_k, d_v, emsize, nhead, &device),
                    |mut attention| {
                        black_box(attention.forward(
                            black_box(input.clone()),
                            None, false, false, false, false, None, false, false,
                        ))
                    }
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark regression test - ensure no performance degradation
fn benchmark_regression_baseline(c: &mut Criterion) {
    let device = Default::default();
    
    // Use the same config as basic_self_attention test for consistency
    let input = create_test_input(1, 4, 32, &device);
    
    let mut group = c.benchmark_group("regression_baseline");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);
    
    group.bench_function("basic_self_attention", |b| {
        b.iter_with_setup(
            || create_attention(16, 16, 32, 2, &device),
            |mut attention| {
                black_box(attention.forward(
                    black_box(input.clone()),
                    None, false, false, false, false, None, false, false,
                ))
            }
        );
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_attention_configurations,
    benchmark_caching_operations,
    benchmark_state_management,
    benchmark_memory_scaling,
    benchmark_head_scaling,
    benchmark_regression_baseline,
);
criterion_main!(benches);
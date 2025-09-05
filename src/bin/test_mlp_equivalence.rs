//! Test binary for MLP equivalence tests between Python and Rust implementations.
//! Updated for Burn 0.18.0 API compatibility and canonical test case support.
//! This binary reads test data from a JSON file and performs the corresponding MLP operations,
//! outputting results for comparison with Python implementation.

use burn::module::Param;
use burn::tensor::{activation, Tensor};
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use tab_pfn_rs::tabpfn::architectures::base::mlp::{Activation, MLP};
use tab_pfn_rs::tabpfn::architectures::base::transformer::DeterministicRngContext;

type Backend = NdArray<f32>;

#[derive(Deserialize)]
struct TestData {
    test_type: String,
    #[serde(flatten)]
    data: serde_json::Value,
}

// Legacy format support
#[derive(Deserialize)]
struct MLPTestData {
    size: usize,
    hidden_size: usize,
    activation: String,
    input: Vec<Vec<f32>>,
    linear1_weight: Vec<Vec<f32>>,
    linear2_weight: Vec<Vec<f32>>,
    expected_output: Vec<Vec<f32>>,
    add_input: bool,
    allow_inplace: bool,
    save_peak_mem_factor: Option<usize>,
}

// Canonical fixture format
#[derive(Deserialize)]
struct CanonicalMLPData {
    input: Vec<f32>,
    network: NetworkConfig,
    weights: WeightData,
    expected_output: Vec<f32>,
    tolerance: Tolerance,
}

#[derive(Deserialize)]
struct NetworkConfig {
    size: usize,
    hidden_size: usize,
    activation: String,
    bias: bool,
}

#[derive(Deserialize)]
struct WeightData {
    linear1: Vec<Vec<f32>>, // PyTorch layout [out_features, in_features]
    linear2: Vec<Vec<f32>>, // PyTorch layout [out_features, in_features]
}

#[derive(Deserialize)]
struct Tolerance {
    atol: f64,
    rtol: f64,
}

#[derive(Deserialize)]
struct ActivationTestData {
    input: Vec<Vec<f32>>,
    expected_gelu: Vec<Vec<f32>>,
    expected_relu: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct TestResult {
    test_type: String,
    success: bool,
    rust_output: Option<Vec<f32>>,
    error: Option<String>,
    max_diff: Option<f32>,
    details: Option<String>,
}

fn test_activation_functions(data: ActivationTestData) -> TestResult {
    let actual_device = Default::default();
    let device = DeterministicRngContext::<Backend>::new(42, actual_device);
    
    // Flatten input data for tensor creation
    let input_flat: Vec<f32> = data.input.into_iter().flatten().collect();
    let batch_size = 1;
    let feature_size = input_flat.len();
    
    // Convert input to tensor
    let input_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(input_flat.as_slice(), device.device())
        .reshape([batch_size, feature_size]);
    
    // Test GELU
    let rust_gelu = activation::gelu(input_tensor.clone());
    let gelu_data: Vec<f32> = rust_gelu.into_data().to_vec().unwrap();
    
    // Test RELU
    let rust_relu = activation::relu(input_tensor);
    let relu_data: Vec<f32> = rust_relu.into_data().to_vec().unwrap();
    
    // Calculate differences
    let gelu_expected: Vec<f32> = data.expected_gelu.into_iter().flatten().collect();
    let relu_expected: Vec<f32> = data.expected_relu.into_iter().flatten().collect();
    
    let gelu_diff = calculate_max_difference_flat(&gelu_data, &gelu_expected);
    let relu_diff = calculate_max_difference_flat(&relu_data, &relu_expected);
    
    let max_diff = gelu_diff.max(relu_diff);
    let success = max_diff < 1e-6;
    
    TestResult {
        test_type: "activation".to_string(),
        success,
        rust_output: Some(gelu_data),
        error: None,
        max_diff: Some(max_diff),
        details: Some(format!("GELU diff: {}, RELU diff: {}", gelu_diff, relu_diff)),
    }
}

fn test_canonical_mlp(data: CanonicalMLPData) -> TestResult {
    let actual_device = Default::default();
    let device = DeterministicRngContext::<Backend>::new(42, actual_device);
    
    // Parse activation
    let activation = match Activation::from_str(&data.network.activation) {
        Ok(act) => act,
        Err(e) => return TestResult {
            test_type: "mlp_canonical".to_string(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
            details: None,
        }
    };
    
    // Create MLP
    let (mut mlp, config) = MLP::<Backend>::new(
        data.network.size,
        data.network.hidden_size,
        activation,
        &device,
        0, // init_seed_offset
        false, // initialize_output_to_zero
        false, // recompute
    );
    
    // Set weights from canonical fixture
    if let Err(e) = set_mlp_weights_canonical(&mut mlp, &data.weights, device.device()) {
        return TestResult {
            test_type: "mlp_canonical".to_string(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
            details: None,
        };
    }
    
    // Convert input to tensor
    let input_tensor: Tensor<Backend, 1> = 
        Tensor::<Backend, 1>::from_floats(data.input.as_slice(), device.device());
    
    // Forward pass
    let output = mlp.mlp_forward(
        input_tensor,
        &config,
        false, // add_input
        false, // allow_inplace
        None,  // save_peak_mem_factor
    );
    
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    let max_diff = calculate_max_difference_flat(&output_data, &data.expected_output);
    let success = max_diff < data.tolerance.atol as f32;
    
    TestResult {
        test_type: "mlp_canonical".to_string(),
        success,
        rust_output: Some(output_data.clone()),
        error: None,
        max_diff: Some(max_diff),
        details: Some(format!(
            "Tolerance: atol={}, rtol={}, actual_diff={}", 
            data.tolerance.atol, data.tolerance.rtol, max_diff
        )),
    }
}

fn test_mlp_forward(data: MLPTestData, test_type: String) -> TestResult {
    let actual_device = Default::default();
    let device = DeterministicRngContext::<Backend>::new(42, actual_device);
    
    // Parse activation
    let activation = match Activation::from_str(&data.activation) {
        Ok(act) => act,
        Err(e) => return TestResult {
            test_type: test_type.clone(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
            details: None,
        }
    };
    
    // Create MLP
    let (mut mlp, config) = MLP::<Backend>::new(
        data.size,
        data.hidden_size,
        activation,
        &device,
        0, // init_seed_offset
        false, // initialize_output_to_zero
        false, // recompute
    );
    
    // Set weights from Python model
    if let Err(e) = set_mlp_weights_legacy(&mut mlp, &data.linear1_weight, &data.linear2_weight, device.device()) {
        return TestResult {
            test_type: "mlp_forward".to_string(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
            details: None,
        };
    }
    
    // Convert input to tensor
    let batch_size = data.input.len();
    let feature_size = data.input[0].len();
    let input_flat: Vec<f32> = data.input.into_iter().flatten().collect();
    let input_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(input_flat.as_slice(), device.device())
        .reshape([batch_size, feature_size]);
    
    // Forward pass
    let output = mlp.mlp_forward(
        input_tensor,
        &config,
        data.add_input,
        data.allow_inplace,
        data.save_peak_mem_factor,
    );
    
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    let expected_flat: Vec<f32> = data.expected_output.into_iter().flatten().collect();
    let max_diff = calculate_max_difference_flat(&output_data, &expected_flat);
    let success = max_diff < 1e-5;
    
    TestResult {
        test_type: test_type.clone(),
        success,
        rust_output: Some(output_data),
        error: None,
        max_diff: Some(max_diff),
        details: Some(format!("Legacy MLP test with tolerance 1e-5")),
    }
}

fn set_mlp_weights_canonical(
    mlp: &mut MLP<Backend>,
    weights: &WeightData,
    device: &<Backend as burn::prelude::Backend>::Device,
) -> Result<(), String> {
    // Convert PyTorch layout [out_features, in_features] to Burn layout [in_features, out_features]
    
    // Linear1 weights
    let linear1_burn_weights = pytorch_to_burn_weights(&weights.linear1);
    let in_features = weights.linear1[0].len();
    let out_features = weights.linear1.len();
    
    let linear1_tensor: Tensor<Backend, 2> = 
        Tensor::<Backend, 1>::from_floats(linear1_burn_weights.as_slice(), device)
        .reshape([in_features, out_features]); 
    mlp.linear1.weight = Param::from_tensor(linear1_tensor);
    
    // Linear2 weights  
    let linear2_burn_weights = pytorch_to_burn_weights(&weights.linear2);
    let in_features2 = weights.linear2[0].len(); 
    let out_features2 = weights.linear2.len();
    
    let linear2_tensor: Tensor<Backend, 2> = 
        Tensor::<Backend, 1>::from_floats(linear2_burn_weights.as_slice(), device)
        .reshape([in_features2, out_features2]);
    mlp.linear2.weight = Param::from_tensor(linear2_tensor);
    
    Ok(())
}

fn set_mlp_weights_legacy(
    mlp: &mut MLP<Backend>,
    linear1_weight: &[Vec<f32>],
    linear2_weight: &[Vec<f32>],
    device: &<Backend as burn::prelude::Backend>::Device,
) -> Result<(), String> {
    // Legacy weight setting with transpose - maintain backward compatibility
    
    // Set linear1 weights - transpose from PyTorch to Burn layout
    let linear1_rows = linear1_weight.len();    
    let linear1_cols = linear1_weight[0].len(); 
    let mut linear1_transposed = vec![vec![0.0f32; linear1_rows]; linear1_cols];
    
    for i in 0..linear1_rows {
        for j in 0..linear1_cols {
            linear1_transposed[j][i] = linear1_weight[i][j];
        }
    }
    
    let linear1_flat: Vec<f32> = linear1_transposed.iter().flatten().cloned().collect();
    let linear1_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(linear1_flat.as_slice(), device)
        .reshape([linear1_cols, linear1_rows]); 
    mlp.linear1.weight = Param::from_tensor(linear1_tensor);
    
    // Set linear2 weights - transpose from PyTorch to Burn layout
    let linear2_rows = linear2_weight.len();    
    let linear2_cols = linear2_weight[0].len(); 
    let mut linear2_transposed = vec![vec![0.0f32; linear2_rows]; linear2_cols];
    
    for i in 0..linear2_rows {
        for j in 0..linear2_cols {
            linear2_transposed[j][i] = linear2_weight[i][j];
        }
    }
    
    let linear2_flat: Vec<f32> = linear2_transposed.iter().flatten().cloned().collect();
    let linear2_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(linear2_flat.as_slice(), device)
        .reshape([linear2_cols, linear2_rows]);  
    mlp.linear2.weight = Param::from_tensor(linear2_tensor);
    
    Ok(())
}

fn pytorch_to_burn_weights(pytorch_weights: &[Vec<f32>]) -> Vec<f32> {
    // Convert PyTorch layout [out_features, in_features] to Burn layout [in_features, out_features]
    let out_features = pytorch_weights.len();
    let in_features = pytorch_weights[0].len();
    
    let mut burn_weights = vec![0.0f32; out_features * in_features];
    
    for i in 0..in_features {
        for j in 0..out_features {
            burn_weights[i * out_features + j] = pytorch_weights[j][i];
        }
    }
    
    burn_weights
}

fn calculate_max_difference_flat(output1: &[f32], output2: &[f32]) -> f32 {
    let mut max_diff = 0.0f32;
    
    for (val1, val2) in output1.iter().zip(output2.iter()) {
        let diff = (val1 - val2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    
    max_diff
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <test_data.json>", args[0]);
        std::process::exit(1);
    }
    
    let test_file = &args[1];
    let test_data_str = match fs::read_to_string(test_file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading test file: {}", e);
            std::process::exit(1);
        }
    };
    
    let test_data: TestData = match serde_json::from_str(&test_data_str) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error parsing test data: {}", e);
            std::process::exit(1);
        }
    };
    
    let result = match test_data.test_type.as_str() {
        "activation" => {
            match serde_json::from_value(test_data.data) {
                Ok(activation_data) => {
                    let activation_data: ActivationTestData = activation_data;
                    if activation_data.input.is_empty() {
                        TestResult {
                            test_type: "activation".to_string(),
                            success: false,
                            rust_output: None,
                            error: Some("Empty input data".to_string()),
                            max_diff: None,
                            details: None,
                        }
                    } else {
                        test_activation_functions(activation_data)
                    }
                },
                Err(e) => TestResult {
                    test_type: "activation".to_string(),
                    success: false,
                    rust_output: None,
                    error: Some(format!("Error parsing activation test data: {}", e)),
                    max_diff: None,
                    details: None,
                }
            }
        },
        "mlp_canonical" => {
            let canonical_data: CanonicalMLPData = match serde_json::from_value(test_data.data) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error parsing canonical MLP test data: {}", e);
                    std::process::exit(1);
                }
            };
            test_canonical_mlp(canonical_data)
        },
        "mlp_forward" | "mlp_memory_opt" => {
            let mlp_data: MLPTestData = match serde_json::from_value(test_data.data) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error parsing MLP test data: {}", e);
                    std::process::exit(1);
                }
            };
            test_mlp_forward(mlp_data, test_data.test_type.clone())
        },
        _ => TestResult {
            test_type: test_data.test_type.clone(),
            success: false,
            rust_output: None,
            error: Some(format!("Unknown test type: {}", test_data.test_type)),
            max_diff: None,
            details: None,
        }
    };
    
    // Output result as JSON
    match serde_json::to_string(&result) {
        Ok(json_output) => println!("{}", json_output),
        Err(e) => {
            eprintln!("Error serializing result: {}", e);
            std::process::exit(1);
        }
    }
}
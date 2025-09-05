//! Test binary for MLP equivalence tests between Python and Rust implementations.
//! This binary reads test data from a JSON file and performs the corresponding MLP operations,
//! outputting results for comparison with Python implementation.

use burn::prelude::*;
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use tab_pfn_rs::tabpfn::architectures::base::mlp::{Activation, MLP};

type Backend = NdArray<f32>;

#[derive(Deserialize)]
struct TestData {
    test_type: String,
    #[serde(flatten)]
    data: serde_json::Value,
}

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
    rust_output: Option<Vec<Vec<f32>>>,
    error: Option<String>,
    max_diff: Option<f32>,
}

fn test_activation_functions(data: ActivationTestData) -> TestResult {
    let device = Default::default();
    
    // Flatten input data for tensor creation
    let input_flat: Vec<f32> = data.input.into_iter().flatten().collect();
    let batch_size = 1;
    let feature_size = input_flat.len();
    
    // Convert input to tensor
    let input_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(input_flat.as_slice(), &device)
        .reshape([batch_size, feature_size]);
    
    // Test GELU
    let rust_gelu = burn::tensor::activation::gelu(input_tensor.clone());
    let gelu_data = rust_gelu.to_data();
    let gelu_output = vec![gelu_data.as_slice::<f32>().unwrap().to_vec()];
    
    // Test RELU
    let rust_relu = burn::tensor::activation::relu(input_tensor);
    let relu_data = rust_relu.to_data();
    let relu_output = vec![relu_data.as_slice::<f32>().unwrap().to_vec()];
    
    // Calculate differences
    let gelu_diff = calculate_max_difference(&gelu_output, &data.expected_gelu);
    let relu_diff = calculate_max_difference(&relu_output, &data.expected_relu);
    
    let max_diff = gelu_diff.max(relu_diff);
    let success = max_diff < 1e-6;
    
    TestResult {
        test_type: "activation".to_string(),
        success,
        rust_output: Some(vec![gelu_output[0].clone(), relu_output[0].clone()]),
        error: None,
        max_diff: Some(max_diff),
    }
}

fn test_mlp_forward(data: MLPTestData, test_type: String) -> TestResult {
    let device = Default::default();
    
    // Parse activation
    let activation = match Activation::from_str(&data.activation) {
        Ok(act) => act,
        Err(e) => return TestResult {
            test_type: test_type.clone(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
        }
    };
    
    // Create MLP
    let (mut mlp, config) = MLP::<Backend>::new(
        data.size,
        data.hidden_size,
        activation,
        &device,
        false, // initialize_output_to_zero
        false, // recompute
    );
    
    // Set weights from Python model
    if let Err(e) = set_mlp_weights(&mut mlp, &data.linear1_weight, &data.linear2_weight, &device) {
        return TestResult {
            test_type: "mlp_forward".to_string(),
            success: false,
            rust_output: None,
            error: Some(e),
            max_diff: None,
        };
    }
    
    // Convert input to tensor
    let batch_size = data.input.len();
    let feature_size = data.input[0].len();
    let input_flat: Vec<f32> = data.input.into_iter().flatten().collect();
    let input_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(input_flat.as_slice(), &device)
        .reshape([batch_size, feature_size]);
    
    // Forward pass
    let output = mlp.mlp_forward(
        input_tensor,
        &config,
        data.add_input,
        data.allow_inplace,
        data.save_peak_mem_factor,
    );
    
    let output_data = output.to_data();
    let output_flat = output_data.as_slice::<f32>().unwrap();
    let rust_output: Vec<Vec<f32>> = output_flat
        .chunks(feature_size)
        .map(|chunk| chunk.to_vec())
        .collect();
    let max_diff = calculate_max_difference(&rust_output, &data.expected_output);
    let success = max_diff < 1e-5;
    
    TestResult {
        test_type: test_type.clone(),
        success,
        rust_output: Some(rust_output),
        error: None,
        max_diff: Some(max_diff),
    }
}

fn set_mlp_weights(
    mlp: &mut MLP<Backend>,
    linear1_weight: &[Vec<f32>],
    linear2_weight: &[Vec<f32>],
    device: &<Backend as burn::prelude::Backend>::Device,
) -> Result<(), String> {
    // PyTorch weight matrices are [output_size, input_size], but Burn expects [input_size, output_size]
    // So we need to transpose the PyTorch weights
    
    // Set linear1 weights - transpose from PyTorch [8, 4] to Burn [4, 8]
    let linear1_rows = linear1_weight.len();    // 8 (hidden_size)
    let linear1_cols = linear1_weight[0].len(); // 4 (input_size)
    let mut linear1_transposed = vec![vec![0.0f32; linear1_rows]; linear1_cols];
    
    for i in 0..linear1_rows {
        for j in 0..linear1_cols {
            linear1_transposed[j][i] = linear1_weight[i][j];
        }
    }
    
    let linear1_flat: Vec<f32> = linear1_transposed.iter().flatten().cloned().collect();
    let linear1_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(linear1_flat.as_slice(), device)
        .reshape([linear1_cols, linear1_rows]); // [4, 8]
    mlp.linear1.weight = burn::module::Param::from_tensor(linear1_tensor);
    
    // Set linear2 weights - transpose from PyTorch [4, 8] to Burn [8, 4]
    let linear2_rows = linear2_weight.len();    // 4 (output_size)
    let linear2_cols = linear2_weight[0].len(); // 8 (hidden_size)
    let mut linear2_transposed = vec![vec![0.0f32; linear2_rows]; linear2_cols];
    
    for i in 0..linear2_rows {
        for j in 0..linear2_cols {
            linear2_transposed[j][i] = linear2_weight[i][j];
        }
    }
    
    let linear2_flat: Vec<f32> = linear2_transposed.iter().flatten().cloned().collect();
    let linear2_tensor: Tensor<Backend, 2> = Tensor::<Backend, 1>::from_floats(linear2_flat.as_slice(), device)
        .reshape([linear2_cols, linear2_rows]); // [8, 4]  
    mlp.linear2.weight = burn::module::Param::from_tensor(linear2_tensor);
    
    Ok(())
}

fn calculate_max_difference(output1: &[Vec<f32>], output2: &[Vec<f32>]) -> f32 {
    let mut max_diff = 0.0f32;
    
    for (row1, row2) in output1.iter().zip(output2.iter()) {
        for (val1, val2) in row1.iter().zip(row2.iter()) {
            let diff = (val1 - val2).abs();
            if diff > max_diff {
                max_diff = diff;
            }
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
            let activation_data: ActivationTestData = match serde_json::from_value(test_data.data) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("Error parsing activation test data: {}", e);
                    std::process::exit(1);
                }
            };
            test_activation_functions(activation_data)
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
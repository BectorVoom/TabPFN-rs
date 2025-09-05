//! Canonical MLP fixture test for cross-language verification
//! 
//! This test uses the exact canonical values specified in the user requirements
//! to ensure numerical parity between Python and Rust implementations.

use burn::module::Param;
use burn::nn::LinearConfig;
use burn::tensor::{activation, Tensor};
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};
use std::fs;

use tab_pfn_rs::tabpfn::architectures::base::mlp::{Activation, MLP, MLPConfig};

type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

#[derive(Deserialize, Debug)]
struct CanonicalFixture {
    test_type: String,
    description: String,
    input_shape: Vec<usize>,
    input: Vec<f32>,
    network: NetworkConfig,
    weights: WeightData,
    expected_output: Vec<f32>,
    tolerance: Tolerance,
    intermediate_values: IntermediateValues,
    metadata: Metadata,
}

#[derive(Deserialize, Debug)]
struct NetworkConfig {
    size: usize,
    hidden_size: usize,
    activation: String,
    bias: bool,
}

#[derive(Deserialize, Debug)]
struct WeightData {
    linear1: Vec<Vec<f32>>, // PyTorch layout [out_features, in_features]
    linear2: Vec<Vec<f32>>, // PyTorch layout [out_features, in_features]
}

#[derive(Deserialize, Debug)]
struct Tolerance {
    atol: f64,
    rtol: f64,
}

#[derive(Deserialize, Debug)]
struct IntermediateValues {
    hidden_after_linear1: Vec<f32>,
    hidden_after_gelu: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct Metadata {
    gelu_formula: String,
    weight_layout: String,
    note: String,
}

fn load_canonical_fixture() -> CanonicalFixture {
    let fixture_path = "fixtures/mlp_canonical.json";
    let fixture_content = fs::read_to_string(fixture_path)
        .expect("Failed to read canonical fixture file");
    
    serde_json::from_str(&fixture_content)
        .expect("Failed to parse canonical fixture JSON")
}

fn pytorch_to_burn_weights(pytorch_weights: &[Vec<f32>]) -> Vec<f32> {
    // Convert PyTorch layout [out_features, in_features] to Burn layout [in_features, out_features]
    // This requires transposing the matrix
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

#[test]
fn test_canonical_mlp_fixture() {
    println!("🧪 Testing canonical MLP fixture for cross-language verification");
    
    // Load the canonical fixture
    let fixture = load_canonical_fixture();
    println!("📋 Loaded fixture: {}", fixture.description);
    
    // Verify this is the canonical test case
    assert_eq!(fixture.test_type, "mlp_canonical");
    assert_eq!(fixture.input, vec![1.0, 2.0, -1.0]);
    assert_eq!(fixture.network.size, 3);
    assert_eq!(fixture.network.hidden_size, 4);
    assert_eq!(fixture.network.activation, "GELU");
    assert!(!fixture.network.bias);
    
    let device = Default::default();
    
    // Create MLP with exact architecture from fixture
    let (mut mlp, config) = MLP::<TestBackend>::new(
        fixture.network.size,
        fixture.network.hidden_size,
        Activation::GELU,
        &device,
        false, // initialize_output_to_zero
        false, // recompute
    );
    
    // Set exact weights from fixture
    // Convert PyTorch layout to Burn layout and create tensors
    
    // Linear1: PyTorch [4, 3] -> Burn [3, 4]
    let linear1_burn_weights = pytorch_to_burn_weights(&fixture.weights.linear1);
    let linear1_tensor: Tensor<TestBackend, 2> = 
        Tensor::<TestBackend, 1>::from_floats(linear1_burn_weights.as_slice(), &device)
        .reshape([3, 4]); // Burn layout: [in_features, out_features]
    mlp.linear1.weight = Param::from_tensor(linear1_tensor);
    
    // Linear2: PyTorch [3, 4] -> Burn [4, 3]  
    let linear2_burn_weights = pytorch_to_burn_weights(&fixture.weights.linear2);
    let linear2_tensor: Tensor<TestBackend, 2> = 
        Tensor::<TestBackend, 1>::from_floats(linear2_burn_weights.as_slice(), &device)
        .reshape([4, 3]); // Burn layout: [in_features, out_features]
    mlp.linear2.weight = Param::from_tensor(linear2_tensor);
    
    // Create input tensor
    let input_tensor: Tensor<TestBackend, 1> = 
        Tensor::<TestBackend, 1>::from_floats(fixture.input.as_slice(), &device);
    
    println!("🔍 Forward pass verification:");
    
    // Step-by-step verification to match the canonical computation
    
    // Step 1: Linear1 forward
    let hidden = mlp.linear1.forward(input_tensor.clone());
    let hidden_data: Vec<f32> = hidden.clone().into_data().to_vec().unwrap();
    
    println!("   Hidden after linear1: {:?}", hidden_data);
    println!("   Expected:             {:?}", fixture.intermediate_values.hidden_after_linear1);
    
    for (i, (&actual, &expected)) in hidden_data.iter()
        .zip(fixture.intermediate_values.hidden_after_linear1.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-6, 
               "Hidden[{}] mismatch: actual={}, expected={}, diff={}", 
               i, actual, expected, diff);
    }
    println!("   ✅ Linear1 output matches expected values");
    
    // Step 2: GELU activation
    let activated = activation::gelu(hidden);
    let activated_data: Vec<f32> = activated.clone().into_data().to_vec().unwrap();
    
    println!("   Hidden after GELU:   {:?}", activated_data);
    println!("   Expected:            {:?}", fixture.intermediate_values.hidden_after_gelu);
    
    for (i, (&actual, &expected)) in activated_data.iter()
        .zip(fixture.intermediate_values.hidden_after_gelu.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, 
               "GELU[{}] mismatch: actual={}, expected={}, diff={}", 
               i, actual, expected, diff);
    }
    println!("   ✅ GELU output matches expected values");
    
    // Step 3: Linear2 forward  
    let output = mlp.linear2.forward(activated);
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    
    println!("   Final output:        {:?}", output_data);
    println!("   Expected:            {:?}", fixture.expected_output);
    
    for (i, (&actual, &expected)) in output_data.iter().zip(fixture.expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < fixture.tolerance.atol as f32, 
               "Output[{}] mismatch: actual={}, expected={}, diff={}", 
               i, actual, expected, diff);
    }
    println!("   ✅ Final output matches expected values");
    
    // Full MLP forward pass test
    println!("🚀 Testing full MLP forward pass:");
    let mlp_output = mlp.mlp_forward(input_tensor, &config, false, false, None);
    let mlp_output_data: Vec<f32> = mlp_output.into_data().to_vec().unwrap();
    
    println!("   MLP output:          {:?}", mlp_output_data);
    println!("   Expected:            {:?}", fixture.expected_output);
    
    for (i, (&actual, &expected)) in mlp_output_data.iter().zip(fixture.expected_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < fixture.tolerance.atol as f32, 
               "MLP Output[{}] mismatch: actual={}, expected={}, diff={}", 
               i, actual, expected, diff);
    }
    
    println!("🎉 All canonical MLP tests passed!");
    println!("🔗 Cross-language numerical parity verified!");
}

#[test]
fn test_canonical_values_verification() {
    println!("🧮 Verifying canonical values match user specification");
    
    let fixture = load_canonical_fixture();
    
    // Verify exact canonical values from user specification
    let expected_canonical_output = vec![0.11585194f32, -0.13783130f32, 0.84134475f32];
    
    for (i, (&actual, &expected)) in fixture.expected_output.iter()
        .zip(expected_canonical_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        println!("   Canonical[{}]: actual={}, expected={}, diff={}", 
                i, actual, expected, diff);
        assert!(diff < 1e-6, 
               "Canonical value[{}] mismatch: actual={}, expected={}", 
               i, actual, expected);
    }
    
    println!("✅ Canonical values match user specification exactly");
}
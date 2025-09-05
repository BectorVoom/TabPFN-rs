//! Verification test for Burn 0.18.0 API compatibility
//! Tests the blocking conditions mentioned in the plan

use burn::module::Param;
use burn::nn::LinearConfig;
use burn::tensor::{activation, Tensor};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_burn_api_compatibility() {
    let device = Default::default();
    
    // Test 1: LinearConfig::new API and with_bias
    let linear1: burn::nn::Linear<TestBackend> = LinearConfig::new(3, 4)
        .with_bias(false)
        .init(&device);
    
    let linear2: burn::nn::Linear<TestBackend> = LinearConfig::new(4, 3)
        .with_bias(false)
        .init(&device);
    
    // Test 2: Linear weight access as Param<Tensor<B,2>>
    let weight1_shape = linear1.weight.val().shape();
    let weight2_shape = linear2.weight.val().shape();
    
    // Verify weight shapes (this tells us about the layout convention)
    assert_eq!(weight1_shape.dims, [3, 4], "linear1 weight should be [in_features, out_features] in Burn");
    assert_eq!(weight2_shape.dims, [4, 3], "linear2 weight should be [in_features, out_features] in Burn");
    
    // Test 3: Tensor creation from arrays
    let input_data = vec![1.0f32, 2.0f32, -1.0f32];
    let input_tensor: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device);
    assert_eq!(input_tensor.shape().dims, [3]);
    
    // Test 4: Weight tensor creation and assignment
    let w1_data = [
        [0.1f32, 0.2f32, 0.3f32],
        [0.0f32, -0.1f32, 0.2f32],
        [0.5f32, 0.5f32, 0.5f32],
        [-0.2f32, 0.1f32, 0.0f32],
    ];
    
    // Create tensor from nested array (transposed for Burn layout)
    let w1_flat: Vec<f32> = w1_data.iter().flatten().copied().collect();
    let w1_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(w1_flat.as_slice(), &device)
        .reshape([4, 3])
        .transpose(); // Transpose to match Burn's [in_features, out_features] layout
    
    // Test 5: GELU activation function
    let test_vals = vec![0.2f32, -0.4f32, 1.0f32, 0.0f32];
    let test_tensor: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats(test_vals.as_slice(), &device);
    let gelu_output = activation::gelu(test_tensor);
    
    // Extract GELU values to check if they match expected exact erf formula
    let gelu_values: Vec<f32> = gelu_output.into_data().to_vec().unwrap();
    
    // These are the expected values using exact erf GELU: x * 0.5 * (1 + erf(x/√2))
    // If Burn uses tanh approximation, these will be different
    let expected_exact_gelu = [0.11585194f32, -0.13783130f32, 0.84134475f32, 0.0f32];
    
    println!("GELU values from Burn: {:?}", gelu_values);
    println!("Expected exact erf GELU: {:?}", expected_exact_gelu);
    
    // Check if values are close to exact erf GELU (tolerance for potential precision differences)
    for (i, (&actual, &expected)) in gelu_values.iter().zip(expected_exact_gelu.iter()).enumerate() {
        let diff = (actual - expected).abs();
        println!("GELU({}) diff: {} (actual: {}, expected: {})", 
                 test_vals[i], diff, actual, expected);
        assert!(diff < 0.01, "GELU value {} differs too much from exact erf formula", i);
    }
}

#[test]
fn test_weight_setting_api() {
    let device = Default::default();
    let mut linear: burn::nn::Linear<TestBackend> = LinearConfig::new(3, 4)
        .with_bias(false)
        .init(&device);
    
    // Test setting weights via Param::from_tensor (Burn layout: [in_features, out_features])
    let new_weight_data: Vec<f32> = (0..12).map(|x| x as f32 * 0.1).collect();
    let new_weight: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(new_weight_data.as_slice(), &device)
        .reshape([3, 4]); // Burn layout: [3 in_features, 4 out_features]
    
    linear.weight = Param::from_tensor(new_weight.clone());
    
    // Verify the weight was set correctly
    let current_weight = linear.weight.val();
    let weight_data: Vec<f32> = current_weight.clone().into_data().to_vec().unwrap();
    
    assert_eq!(weight_data.len(), 12);
    assert!((weight_data[0] - 0.0).abs() < 1e-6);
    assert!((weight_data[1] - 0.1).abs() < 1e-6);
    assert!((weight_data[2] - 0.2).abs() < 1e-6);
}
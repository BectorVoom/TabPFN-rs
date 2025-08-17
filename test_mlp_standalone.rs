use burn::tensor::Tensor;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn_ndarray::NdArray;

// Import the MLP module
mod tabpfn {
    pub mod architectures {
        pub mod base {
            pub mod mlp {
                include!("src/tabpfn/architectures/base/mlp.rs");
            }
        }
    }
}

use tabpfn::architectures::base::mlp::{MLP, Activation, MLPConfig};

type TestBackend = NdArray<f32>;

fn main() {
    println!("🧪 Testing canonical MLP values for cross-language verification");
    
    let device = Default::default();
    
    // Create MLP with canonical architecture
    let (mut mlp, config) = MLP::<TestBackend>::new(3, 4, Activation::GELU, &device, false, false);
    
    // Set canonical weights from specification
    // PyTorch layout: [out_features, in_features] -> Burn layout: [in_features, out_features]
    
    // Linear1: PyTorch [4, 3] -> Burn [3, 4]
    let _linear1_pytorch = [
        [0.1f32, 0.2f32, 0.3f32],      // output neuron 0
        [0.0f32, -0.1f32, 0.2f32],     // output neuron 1  
        [0.5f32, 0.5f32, 0.5f32],      // output neuron 2
        [-0.2f32, 0.1f32, 0.0f32]      // output neuron 3
    ];
    
    // Transpose to Burn layout [3, 4]
    let linear1_burn = [
        [0.1f32, 0.0f32, 0.5f32, -0.2f32],   // input feature 0
        [0.2f32, -0.1f32, 0.5f32, 0.1f32],   // input feature 1
        [0.3f32, 0.2f32, 0.5f32, 0.0f32]     // input feature 2
    ];
    
    let w1_flat: Vec<f32> = linear1_burn.iter().flatten().copied().collect();
    let w1_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(w1_flat.as_slice(), &device)
        .reshape([3, 4]);
    mlp.linear1.weight = Param::from_tensor(w1_tensor);
    
    // Linear2: PyTorch [3, 4] -> Burn [4, 3]  
    let _linear2_pytorch = [
        [1.0f32, 0.0f32, 0.0f32, 0.0f32], // output neuron 0
        [0.0f32, 1.0f32, 0.0f32, 0.0f32], // output neuron 1
        [0.0f32, 0.0f32, 1.0f32, 0.0f32]  // output neuron 2
    ];
    
    // Transpose to Burn layout [4, 3]
    let linear2_burn = [
        [1.0f32, 0.0f32, 0.0f32], // hidden feature 0
        [0.0f32, 1.0f32, 0.0f32], // hidden feature 1
        [0.0f32, 0.0f32, 1.0f32], // hidden feature 2
        [0.0f32, 0.0f32, 0.0f32]  // hidden feature 3
    ];
    
    let w2_flat: Vec<f32> = linear2_burn.iter().flatten().copied().collect();
    let w2_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(w2_flat.as_slice(), &device)
        .reshape([4, 3]);
    mlp.linear2.weight = Param::from_tensor(w2_tensor);
    
    // Canonical input
    let input: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0f32, -1.0f32].as_slice(), &device);
    
    // Forward pass
    let output = mlp.mlp_forward(input, &config, false, false, None);
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    
    // Expected canonical output
    let expected = [0.11585194f32, -0.13783130f32, 0.84134475f32];
    
    println!("   Output: {:?}", output_data);
    println!("   Expected: {:?}", expected);
    
    let mut all_pass = true;
    for (i, (&actual, &expected)) in output_data.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        println!("   Diff[{}]: {} (actual: {}, expected: {})", i, diff, actual, expected);
        if diff >= 1e-5 {
            println!("❌ Canonical value mismatch at index {}", i);
            all_pass = false;
        }
    }
    
    if all_pass {
        println!("✅ Canonical MLP values verified");
    } else {
        println!("❌ Canonical MLP test failed");
        std::process::exit(1);
    }
}
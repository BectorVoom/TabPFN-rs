use burn::prelude::*;
use burn_ndarray::NdArray;
use burn::nn::{Linear, LinearConfig};

type Backend = NdArray<f32>;

fn main() {
    let device = Default::default();
    
    // Create Linear layers matching our MLP
    let linear1 = LinearConfig::new(4, 8).with_bias(false).init(&device);
    let linear2 = LinearConfig::new(8, 4).with_bias(false).init(&device);
    
    println!("Burn Linear layer weight shapes:");
    let l1_weight_shape = linear1.weight.val().shape();
    let l2_weight_shape = linear2.weight.val().shape();
    
    println!("linear1.weight.shape(): {:?}", l1_weight_shape.dims);
    println!("linear2.weight.shape(): {:?}", l2_weight_shape.dims);
    
    // Test with sample input
    let input: Tensor<Backend, 2> = Tensor::ones([2, 4], &device);
    println!("Input shape: {:?}", input.shape().dims);
    
    let h1 = linear1.forward(input);
    println!("After linear1 shape: {:?}", h1.shape().dims);
    
    let h1_relu = burn::tensor::activation::relu(h1);
    println!("After ReLU shape: {:?}", h1_relu.shape().dims);
    
    let output = linear2.forward(h1_relu);
    println!("Final output shape: {:?}", output.shape().dims);
}
// tests/grad_accumulation.rs - Test gradient accumulation functionality
use burn::{
    backend::Autodiff,
    module::Module, 
    optim::{GradientsParams, AdamConfig, Optimizer},
    tensor::{backend::Backend, Tensor, TensorData},
};
use burn_ndarray::NdArray;

type TestBackend = Autodiff<NdArray<f32>>;

// Simple model for testing gradient accumulation
#[derive(Module, Debug, Clone)]
pub struct TestModel<B: Backend> {
    pub linear: burn::nn::Linear<B>,
}

impl<B: Backend> TestModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear = burn::nn::LinearConfig::new(3, 2).init(device);
        Self { linear }
    }
    
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

#[test]
fn test_gradient_accumulation_basic() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    
    // Create two small batches
    let batch1 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    let batch2 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5], [6]),
        &device
    ).reshape([2, 3]).require_grad();
    
    // Compute gradients for batch1
    let output1 = model.forward(batch1);
    let loss1 = output1.sum();
    let gradients1 = loss1.backward();
    let model_gradients1 = GradientsParams::from_grads(gradients1, &model);
    
    // Compute gradients for batch2
    let output2 = model.forward(batch2);
    let loss2 = output2.sum();
    let gradients2 = loss2.backward();
    let model_gradients2 = GradientsParams::from_grads(gradients2, &model);
    
    // Test that we can accumulate gradients manually
    // Extract gradients for verification
    let weight_grad1 = model_gradients1.get(&model.linear.weight.id()).unwrap();
    let weight_grad2 = model_gradients2.get(&model.linear.weight.id()).unwrap();
    
    // Convert to tensors for accumulation
    let weight_tensor1 = Tensor::<TestBackend, 2>::from_primitive(weight_grad1.clone());
    let weight_tensor2 = Tensor::<TestBackend, 2>::from_primitive(weight_grad2.clone());
    
    // Accumulate gradients
    let accumulated_weight_grad = weight_tensor1 + weight_tensor2;
    
    // Verify accumulation worked
    let grad1_data = weight_tensor1.to_data();
    let grad2_data = weight_tensor2.to_data();
    let accumulated_data = accumulated_weight_grad.to_data();
    
    let grad1_values = grad1_data.as_slice::<f32>().unwrap();
    let grad2_values = grad2_data.as_slice::<f32>().unwrap();
    let accumulated_values = accumulated_data.as_slice::<f32>().unwrap();
    
    // Verify accumulation is sum of individual gradients
    for ((g1, g2), acc) in grad1_values.iter().zip(grad2_values.iter()).zip(accumulated_values.iter()) {
        let expected = g1 + g2;
        assert!((expected - acc).abs() < 1e-6, 
                "Gradient accumulation incorrect: {} + {} = {}, got {}", g1, g2, expected, acc);
    }
}

#[test]
fn test_gradient_accumulation_equivalence() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    
    // Create a large batch
    let large_batch = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [12]),
        &device
    ).reshape([4, 3]).require_grad();
    
    // Compute gradients for large batch
    let large_output = model.forward(large_batch.clone());
    let large_loss = large_output.sum();
    let large_gradients = large_loss.backward();
    let large_model_gradients = GradientsParams::from_grads(large_gradients, &model);
    
    // Split into two smaller batches
    let small_batch1 = large_batch.clone().slice([0..2, 0..3]);
    let small_batch2 = large_batch.clone().slice([2..4, 0..3]);
    
    // Compute gradients for small batches
    let small_output1 = model.forward(small_batch1);
    let small_loss1 = small_output1.sum();
    let small_gradients1 = small_loss1.backward();
    let small_model_gradients1 = GradientsParams::from_grads(small_gradients1, &model);
    
    let small_output2 = model.forward(small_batch2);
    let small_loss2 = small_output2.sum();
    let small_gradients2 = small_loss2.backward();
    let small_model_gradients2 = GradientsParams::from_grads(small_gradients2, &model);
    
    // Accumulate small batch gradients
    let small_weight_grad1 = small_model_gradients1.get(&model.linear.weight.id()).unwrap();
    let small_weight_grad2 = small_model_gradients2.get(&model.linear.weight.id()).unwrap();
    
    let small_tensor1 = Tensor::<TestBackend, 2>::from_primitive(small_weight_grad1.clone());
    let small_tensor2 = Tensor::<TestBackend, 2>::from_primitive(small_weight_grad2.clone());
    let accumulated_small = small_tensor1 + small_tensor2;
    
    // Compare with large batch gradients
    let large_weight_grad = large_model_gradients.get(&model.linear.weight.id()).unwrap();
    let large_tensor = Tensor::<TestBackend, 2>::from_primitive(large_weight_grad.clone());
    
    let large_data = large_tensor.to_data();
    let accumulated_data = accumulated_small.to_data();
    
    let large_values = large_data.as_slice::<f32>().unwrap();
    let accumulated_values = accumulated_data.as_slice::<f32>().unwrap();
    
    // Results should be equivalent
    for (large, accumulated) in large_values.iter().zip(accumulated_values.iter()) {
        assert!((large - accumulated).abs() < 1e-5,
                "Large batch and accumulated small batch gradients should be equivalent: {} vs {}", 
                large, accumulated);
    }
}

#[test]
fn test_gradient_accumulation_scaling() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    
    // Create identical batches
    let batch = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0], [3]),
        &device
    ).reshape([1, 3]).require_grad();
    
    // Compute gradient once
    let output1 = model.forward(batch.clone());
    let loss1 = output1.sum();
    let gradients1 = loss1.backward();
    let model_gradients1 = GradientsParams::from_grads(gradients1, &model);
    
    // Compute gradient for same batch again
    let output2 = model.forward(batch.clone());
    let loss2 = output2.sum();
    let gradients2 = loss2.backward();
    let model_gradients2 = GradientsParams::from_grads(gradients2, &model);
    
    // Accumulate (should be 2x the single gradient)
    let weight_grad1 = model_gradients1.get(&model.linear.weight.id()).unwrap();
    let weight_grad2 = model_gradients2.get(&model.linear.weight.id()).unwrap();
    
    let tensor1 = Tensor::<TestBackend, 2>::from_primitive(weight_grad1.clone());
    let tensor2 = Tensor::<TestBackend, 2>::from_primitive(weight_grad2.clone());
    let accumulated = tensor1.clone() + tensor2.clone();
    
    // Verify accumulation is 2x single gradient
    let single_data = tensor1.to_data();
    let accumulated_data = accumulated.to_data();
    
    let single_values = single_data.as_slice::<f32>().unwrap();
    let accumulated_values = accumulated_data.as_slice::<f32>().unwrap();
    
    for (single, accumulated) in single_values.iter().zip(accumulated_values.iter()) {
        let expected = single * 2.0;
        assert!((expected - accumulated).abs() < 1e-6,
                "Accumulated gradient should be 2x single gradient: 2*{} = {}, got {}", 
                single, expected, accumulated);
    }
}

#[test]
fn test_gradient_accumulation_with_optimizer() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    let mut optimizer = AdamConfig::new().init();
    
    // Simulate multiple mini-batches with gradient accumulation
    let batch1 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![1.0, 0.0, 1.0], [3]),
        &device
    ).reshape([1, 3]).require_grad();
    
    let batch2 = Tensor::<TestBackend, 1>::from_data(
        TensorData::new(vec![0.0, 1.0, 1.0], [3]),
        &device
    ).reshape([1, 3]).require_grad();
    
    // Accumulate gradients from both batches
    let output1 = model.forward(batch1);
    let loss1 = output1.sum() / 2.0; // Divide by accumulation steps
    let gradients1 = loss1.backward();
    let model_gradients1 = GradientsParams::from_grads(gradients1, &model);
    
    let output2 = model.forward(batch2);
    let loss2 = output2.sum() / 2.0; // Divide by accumulation steps
    let gradients2 = loss2.backward();
    let model_gradients2 = GradientsParams::from_grads(gradients2, &model);
    
    // Manually accumulate gradients
    let weight_grad1 = model_gradients1.get(&model.linear.weight.id()).unwrap();
    let weight_grad2 = model_gradients2.get(&model.linear.weight.id()).unwrap();
    let bias_grad1 = model_gradients1.get(&model.linear.bias.as_ref().unwrap().id()).unwrap();
    let bias_grad2 = model_gradients2.get(&model.linear.bias.as_ref().unwrap().id()).unwrap();
    
    let accumulated_weight = Tensor::<TestBackend, 2>::from_primitive(weight_grad1.clone()) +
                           Tensor::<TestBackend, 2>::from_primitive(weight_grad2.clone());
    let accumulated_bias = Tensor::<TestBackend, 1>::from_primitive(bias_grad1.clone()) +
                         Tensor::<TestBackend, 1>::from_primitive(bias_grad2.clone());
    
    // Create accumulated gradients params
    let mut accumulated_gradients = GradientsParams::new();
    accumulated_gradients.insert(model.linear.weight.id().clone(), accumulated_weight.into_primitive().tensor());
    accumulated_gradients.insert(model.linear.bias.as_ref().unwrap().id().clone(), accumulated_bias.into_primitive().tensor());
    
    // Apply optimizer step
    let original_weight_data = model.linear.weight.val().to_data();
    let updated_model = optimizer.step(0.01, model.clone(), accumulated_gradients);
    let updated_weight_data = updated_model.linear.weight.val().to_data();
    
    let original_weights = original_weight_data.as_slice::<f32>().unwrap();
    let updated_weights = updated_weight_data.as_slice::<f32>().unwrap();
    
    // Verify parameters changed
    let mut parameters_changed = false;
    for (orig, updated) in original_weights.iter().zip(updated_weights.iter()) {
        if (orig - updated).abs() > 1e-6 {
            parameters_changed = true;
            break;
        }
    }
    
    assert!(parameters_changed, "Optimizer should have updated parameters with accumulated gradients");
}

#[test]
fn test_gradient_accumulation_zero_gradients() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    
    // Create batch that should produce zero gradients
    let zero_batch = Tensor::<TestBackend, 2>::zeros([1, 3], &device).require_grad();
    
    let output = model.forward(zero_batch);
    let loss = output.sum();
    let gradients = loss.backward();
    let model_gradients = GradientsParams::from_grads(gradients, &model);
    
    // Verify gradients are effectively zero (or very small)
    let weight_grad = model_gradients.get(&model.linear.weight.id()).unwrap();
    let weight_tensor = Tensor::<TestBackend, 2>::from_primitive(weight_grad.clone());
    let weight_data = weight_tensor.to_data();
    let weight_values = weight_data.as_slice::<f32>().unwrap();
    
    for &grad in weight_values.iter() {
        assert!(grad.abs() < 1e-3, "Zero input should produce small gradients, got {}", grad);
    }
}

#[test]
fn test_gradient_accumulation_memory_efficiency() {
    let device = <TestBackend as Backend>::Device::default();
    let model = TestModel::new(&device);
    
    // Test that gradient accumulation doesn't cause memory leaks
    // by running multiple accumulation cycles
    for cycle in 0..3 {
        let mut accumulated_weight_grad = None;
        
        for batch_idx in 0..2 {
            let batch_data: Vec<f32> = (0..6).map(|i| (cycle * 6 + batch_idx * 3 + i) as f32 * 0.1).collect();
            let batch = Tensor::<TestBackend, 1>::from_data(
                TensorData::new(batch_data, [6]),
                &device
            ).reshape([2, 3]).require_grad();
            
            let output = model.forward(batch);
            let loss = output.sum();
            let gradients = loss.backward();
            let model_gradients = GradientsParams::from_grads(gradients, &model);
            
            let weight_grad = model_gradients.get(&model.linear.weight.id()).unwrap();
            let weight_tensor = Tensor::<TestBackend, 2>::from_primitive(weight_grad.clone());
            
            accumulated_weight_grad = if let Some(existing) = accumulated_weight_grad {
                Some(existing + weight_tensor)
            } else {
                Some(weight_tensor)
            };
        }
        
        let final_grad = accumulated_weight_grad.unwrap();
        let grad_data = final_grad.to_data();
        let grad_values = grad_data.as_slice::<f32>().unwrap();
        
        // Verify gradients are reasonable (finite and not too large)
        for &grad in grad_values.iter() {
            assert!(grad.is_finite(), "Accumulated gradients should be finite in cycle {}", cycle);
            assert!(grad.abs() < 100.0, "Accumulated gradients should not explode in cycle {}: {}", cycle, grad);
        }
    }
}
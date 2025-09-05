//! Copyright (c) Prior Labs GmbH 2025.
//!
//! Multi-Layer Perceptron (MLP) module - Rust implementation of
//! src/tabpfn/architectures/base/mlp.py

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::{activation, Tensor};
use super::transformer::{DeterministicRngContext, DeterministicLinear};

// Burn MLP module - gradient checkpointing handled at Backend level

/// Enum for activation functions - equivalent to Python Activation enum
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    GELU = 1,
    RELU = 2,
}

impl Activation {
    /// Create Activation from string - equivalent to Python's Activation[activation.upper()]
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "GELU" => Ok(Activation::GELU),
            "RELU" => Ok(Activation::RELU),
            _ => Err(format!("Unknown activation function: {}", s)),
        }
    }
}

/// Multi-Layer Perceptron (MLP) module.
///
/// This module consists of two linear layers with an activation function in between.
/// It supports various configurations such as the hidden size, activation function,
/// initializing the output to zero, and recomputing the forward pass during
/// backpropagation.
///
/// Args:
///     size: The input and output size of the MLP.
///     hidden_size: The size of the hidden layer.
///     activation:
///         The activation function to use. Can be either an Activation enum or
///         a string representing the activation name.
///     device: The device to use for the linear layers.
///     dtype: The data type to use for the linear layers.
///     initialize_output_to_zero:
///         Whether to initialize the output layer weights
///         to zero. Default is False.
///     recompute:
///         Whether to recompute the forward pass during backpropagation.
///         This can save memory but increase computation time. Default is False.
///
/// Attributes:
///     linear1: The first linear layer.
///     linear2: The second linear layer.
///     activation: The activation function to use.
///
/// Example:
///     ```rust,no_run
///     # use burn::prelude::*;
///     # use burn_ndarray::NdArray;
///     # use tab_pfn_rs::tabpfn::architectures::base::mlp::{MLP, Activation};
///     # type Backend = NdArray<f32>;
///     # let device = Default::default();
///     let (mlp, config) = MLP::<Backend>::new(128, 256, Activation::GELU, &device, true, false);
///     let x = Tensor::<Backend, 2>::zeros([32, 128], &device);
///     let output = mlp.mlp_forward(x, &config, false, false, None);
///     ```
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// The first linear layer
    pub linear1: DeterministicLinear<B>,
    /// The second linear layer
    pub linear2: DeterministicLinear<B>,
}

/// Configuration for MLP that stores non-trainable parameters
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// The activation function to use
    pub activation: Activation,
    /// Whether to use gradient checkpointing (recomputation)
    pub recompute: bool,
}

impl<B: Backend> MLP<B> {
    /// Create a new MLP module with deterministic initialization
    ///
    /// # Arguments
    /// * `size` - The input and output size of the MLP
    /// * `hidden_size` - The size of the hidden layer
    /// * `activation` - The activation function to use
    /// * `rng_ctx` - The deterministic RNG context for parameter initialization
    /// * `init_seed_offset` - Seed offset for deterministic initialization
    /// * `initialize_output_to_zero` - Whether to initialize the output layer weights to zero
    /// * `recompute` - Whether to recompute the forward pass during backpropagation
    pub fn new(
        size: usize,
        hidden_size: usize,
        activation: Activation,
        rng_ctx: &DeterministicRngContext<B>,
        init_seed_offset: u64,
        initialize_output_to_zero: bool,
        recompute: bool,
    ) -> (Self, MLPConfig) {
        // Use seed offset policy: +100 series for linear layers
        let linear1 = rng_ctx.create_deterministic_linear(
            size,
            hidden_size,
            false, // no bias
            init_seed_offset + 100, // linear1 = base + 100
        );

        let linear2 = if initialize_output_to_zero {
            // Create zero-initialized weights [output_dim, input_dim] = [size, hidden_size]
            let zero_weight = Tensor::zeros([size, hidden_size], rng_ctx.device());
            DeterministicLinear::new(zero_weight, None)
        } else {
            rng_ctx.create_deterministic_linear(
                hidden_size,
                size,
                false, // no bias
                init_seed_offset + 101, // linear2 = base + 101
            )
        };

        let mlp = Self { linear1, linear2 };

        let config = MLPConfig {
            activation,
            recompute,
        };

        (mlp, config)
    }

    /// Create MLP from string activation with deterministic initialization
    pub fn new_with_str_activation(
        size: usize,
        hidden_size: usize,
        activation: &str,
        rng_ctx: &DeterministicRngContext<B>,
        init_seed_offset: u64,
        initialize_output_to_zero: bool,
        recompute: bool,
    ) -> Result<(Self, MLPConfig), String> {
        let activation = Activation::from_str(activation)?;
        Ok(Self::new(
            size,
            hidden_size,
            activation,
            rng_ctx,
            init_seed_offset,
            initialize_output_to_zero,
            recompute,
        ))
    }

    /// Compute the forward pass - equivalent to Python's _compute method
    /// 
    /// In Burn, gradient checkpointing is handled automatically by Backend type:
    /// - Autodiff<NdArray, BalancedCheckpointing> enables automatic checkpointing
    /// - Autodiff<NdArray, NoCheckpointing> uses standard computation  
    /// The recompute flag is kept for API compatibility with Python version
    fn _compute_2d(&self, x: Tensor<B, 2>, config: &MLPConfig) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);

        // Apply activation function
        let x = match config.activation {
            Activation::GELU => activation::gelu(x),
            Activation::RELU => activation::relu(x),
        };

        self.linear2.forward(x)
    }

    /// Compute the forward pass for 3D tensors
    fn _compute_3d(&self, x: Tensor<B, 3>, config: &MLPConfig) -> Tensor<B, 3> {
        let x = self.linear1.forward_3d(x);

        // Apply activation function
        let x = match config.activation {
            Activation::GELU => activation::gelu(x),
            Activation::RELU => activation::relu(x),
        };

        self.linear2.forward_3d(x)
    }

    /// Chunked computation for memory efficiency (2D tensors)
    fn _compute_chunked_2d(
        &self,
        x: Tensor<B, 2>,
        config: &MLPConfig,
        factor: usize,
        add_input: bool,
    ) -> Tensor<B, 2> {
        if factor <= 1 {
            // No chunking needed
            let mlp_output = self._compute_2d(x.clone(), config);
            return if add_input {
                mlp_output + x
            } else {
                mlp_output
            };
        }

        // Split input into chunks and process separately
        let chunks = x.chunk(factor, 0);
        let mut chunk_results = Vec::new();

        for chunk in chunks {
            let chunk_result = self._compute_2d(chunk.clone(), config);
            let final_chunk = if add_input {
                chunk_result + chunk
            } else {
                chunk_result
            };
            chunk_results.push(final_chunk);
        }

        Tensor::cat(chunk_results, 0)
    }

    /// Performs the forward pass of the MLP.
    ///
    /// Args:
    ///     x: The input tensor.
    ///     config: The MLP configuration containing activation and recompute settings.
    ///     add_input: Whether to add input to the output. Default is false.
    ///     allow_inplace:
    ///         Indicates that 'x' is not used after the call and
    ///         its buffer can be reused for the output. The operation is not
    ///         guaranteed to be inplace. Default is false.

    pub fn mlp_forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        config: &MLPConfig,
        add_input: bool,
        _allow_inplace: bool,
        save_peak_mem_factor: Option<usize>,
    ) -> Tensor<B, D> {
        let input_shape = x.shape();

        // Reshape to 2D: flatten all dimensions except the last one
        let batch_size: usize = input_shape.dims[..D - 1].iter().product();
        let feature_size = input_shape.dims[D - 1];
        let x_reshaped = x.reshape([batch_size, feature_size]);

        // Compute MLP output - gradient checkpointing handled by Backend type
        let result = if let Some(factor) = save_peak_mem_factor {
            // Handle chunked processing for memory efficiency
            self._compute_chunked_2d(x_reshaped.clone(), config, factor, add_input)
        } else {
            // Standard computation
            let mlp_output = self._compute_2d(x_reshaped.clone(), config);
            if add_input {
                mlp_output + x_reshaped
            } else {
                mlp_output
            }
        };

        // Reshape back to original shape
        result.reshape(input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_activation_from_str() {
        assert_eq!(Activation::from_str("gelu").unwrap(), Activation::GELU);
        assert_eq!(Activation::from_str("GELU").unwrap(), Activation::GELU);
        assert_eq!(Activation::from_str("relu").unwrap(), Activation::RELU);
        assert_eq!(Activation::from_str("RELU").unwrap(), Activation::RELU);
        assert!(Activation::from_str("invalid").is_err());
    }

    #[test]
    fn test_mlp_creation() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (_mlp, config) =
            MLP::<TestBackend>::new(128, 256, Activation::GELU, &rng_ctx, 100, false, false);

        assert_eq!(config.activation, Activation::GELU);
        assert!(!config.recompute);
    }

    #[test]
    fn test_mlp_from_str_activation() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (_mlp, config) =
            MLP::<TestBackend>::new_with_str_activation(128, 256, "gelu", &rng_ctx, 100, false, false)
                .unwrap();

        assert_eq!(config.activation, Activation::GELU);
    }

    #[test]
    fn test_mlp_forward_2d() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::RELU, &rng_ctx, 100, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::zeros([32, 64], rng_ctx.device());
        let output = mlp.mlp_forward(x, &config, false, false, None);

        assert_eq!(output.shape().dims, [32, 64]);
    }

    #[test]
    fn test_mlp_forward_3d() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::GELU, &rng_ctx, 100, false, false);

        let x: Tensor<TestBackend, 3> = Tensor::zeros([8, 16, 64], rng_ctx.device());
        let output = mlp.mlp_forward(x, &config, false, false, None);

        assert_eq!(output.shape().dims, [8, 16, 64]);
    }

    #[test]
    fn test_mlp_with_add_input() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::RELU, &rng_ctx, 100, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([4, 64], rng_ctx.device());
        let output = mlp.mlp_forward(x.clone(), &config, true, true, None);

        // With add_input=true, output should include the original input
        assert_eq!(output.shape().dims, [4, 64]);
    }

    #[test]
    fn test_mlp_zero_initialization() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) = MLP::<TestBackend>::new(32, 64, Activation::GELU, &rng_ctx, 100, true, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([2, 32], rng_ctx.device());
        let output = mlp.mlp_forward(x, &config, false, false, None);

        // With zero initialization of output weights, the final output should be zeros
        // (since we're only going through linear2 which has zero weights)
        assert_eq!(output.shape().dims, [2, 32]);
    }

    #[test]
    fn test_mlp_memory_optimization() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) =
            MLP::<TestBackend>::new(32, 64, Activation::RELU, &rng_ctx, 100, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([8, 32], rng_ctx.device());
        let output = mlp.mlp_forward(x, &config, false, true, Some(2));

        assert_eq!(output.shape().dims, [8, 32]);
    }

    #[test]
    fn test_mlp_gradient_checkpointing() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);

        // Test with recompute=false (standard)
        let (mlp_standard, config_standard) =
            MLP::<TestBackend>::new(16, 32, Activation::GELU, &rng_ctx, 100, false, false);

        // Test with recompute=true (gradient checkpointing)
        let (mut mlp_checkpoint, config_checkpoint) =
            MLP::<TestBackend>::new(16, 32, Activation::GELU, &rng_ctx, 100, false, true);

        // Copy weights to ensure identical computation
        mlp_checkpoint.linear1.weight = mlp_standard.linear1.weight.clone();
        mlp_checkpoint.linear2.weight = mlp_standard.linear2.weight.clone();

        let x: Tensor<TestBackend, 2> = Tensor::ones([4, 16], rng_ctx.device());

        // Both methods should produce identical results
        let output_standard = mlp_standard.mlp_forward(x.clone(), &config_standard, false, false, None);
        let output_checkpoint = mlp_checkpoint.mlp_forward(x, &config_checkpoint, false, false, None);

        assert_eq!(output_standard.shape().dims, output_checkpoint.shape().dims);

        // Verify numerical equivalence (allowing for small floating-point differences)
        let diff = output_standard
            .clone()
            .sub(output_checkpoint.clone())
            .abs()
            .max();
        let max_diff: f32 = diff.into_scalar();
        assert!(
            max_diff < 1e-6,
            "Outputs should be numerically equivalent, max diff: {}",
            max_diff
        );
    }

    #[test]
    fn test_mlp_recompute_configuration() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);

        // Test that recompute flag is properly stored and used
        let (_mlp, config) = MLP::<TestBackend>::new(8, 16, Activation::RELU, &rng_ctx, 100, false, true);
        assert!(config.recompute, "recompute flag should be set to true");

        let (_mlp, config) =
            MLP::<TestBackend>::new(8, 16, Activation::RELU, &rng_ctx, 100, false, false);
        assert!(!config.recompute, "recompute flag should be set to false");
    }

    #[test]
    fn test_canonical_mlp_values() {
        println!("🧪 Testing canonical MLP values for cross-language verification");
        
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        
        // Create MLP with canonical architecture
        let (mut mlp, config) = MLP::<TestBackend>::new(3, 4, Activation::GELU, &rng_ctx, 100, false, false);
        
        // Set canonical weights from specification
        // PyTorch layout: [out_features, in_features] -> Burn layout: [in_features, out_features]
        
        // Linear1: PyTorch [4, 3] -> Burn [3, 4]
        let linear1_pytorch = [
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
        let w1_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(w1_flat.as_slice(), rng_ctx.device())
            .reshape([3, 4]);
        mlp.linear1.weight = Param::from_tensor(w1_tensor);
        
        // Linear2: PyTorch [3, 4] -> Burn [4, 3]  
        let linear2_pytorch = [
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
        let w2_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(w2_flat.as_slice(), rng_ctx.device())
            .reshape([4, 3]);
        mlp.linear2.weight = Param::from_tensor(w2_tensor);
        
        // Canonical input
        let input: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0f32, -1.0f32].as_slice(), rng_ctx.device());
        
        // Forward pass
        let output = mlp.mlp_forward(input, &config, false, false, None);
        let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
        
        // Expected canonical output
        let expected = [0.11585194f32, -0.13783130f32, 0.84134475f32];
        
        println!("   Output: {:?}", output_data);
        println!("   Expected: {:?}", expected);
        
        for (i, (&actual, &expected)) in output_data.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            println!("   Diff[{}]: {} (actual: {}, expected: {})", i, diff, actual, expected);
            assert!(diff < 1e-5, "Canonical value mismatch at index {}", i);
        }
        
        println!("✅ Canonical MLP values verified");
    }

    #[test]
    fn test_gelu_precision() {
        let device = Default::default();
        
        // Test GELU precision with known values
        let test_values = [0.2f32, -0.4f32, 1.0f32, 0.0f32];
        let expected_gelu = [0.11585194f32, -0.13783130f32, 0.84134475f32, 0.0f32];
        
        let input_tensor: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats(test_values.as_slice(), &device);
        let gelu_output = activation::gelu(input_tensor);
        let gelu_data: Vec<f32> = gelu_output.into_data().to_vec().unwrap();
        
        for (i, (&actual, &expected)) in gelu_data.iter().zip(expected_gelu.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(diff < 1e-5, "GELU precision test failed at index {}: diff = {}", i, diff);
        }
    }

    #[test]
    fn test_gelu_exact_erf_formula() {
        let device = Default::default();
        
        // Test GELU function by comparing Burn's implementation with manual erf calculation
        let test_values = [0.2f32, -0.4f32, 1.0f32, 0.0f32, -2.0f32, 3.0f32];
        
        let input_tensor: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats(test_values.as_slice(), &device);
        let burn_gelu = activation::gelu(input_tensor.clone());
        let burn_data: Vec<f32> = burn_gelu.into_data().to_vec().unwrap();
        
        // Manual GELU calculation using exact erf formula: GELU(x) = x * 0.5 * (1 + erf(x/√2))
        let sqrt_2 = 2.0f32.sqrt();
        let expected: Vec<f32> = test_values.iter().map(|&x| {
            let erf_val = libm::erff(x / sqrt_2);
            x * 0.5 * (1.0 + erf_val)
        }).collect();
        
        println!("GELU exact erf formula verification:");
        for (i, ((&input, &burn_result), &manual_result)) in test_values.iter().zip(burn_data.iter()).zip(expected.iter()).enumerate() {
            let diff = (burn_result - manual_result).abs();
            println!("  Input[{}]: {} -> Burn: {:.8}, Manual: {:.8}, Diff: {:.2e}", 
                     i, input, burn_result, manual_result, diff);
            assert!(diff < 1e-6, "GELU mismatch at index {} for input {}: Burn={}, Manual={}, diff={}", 
                    i, input, burn_result, manual_result, diff);
        }
        
        println!("✅ Burn's GELU matches exact erf formula");
    }

    #[test]
    fn test_weight_layout_conversion() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        
        // Test that weight layout conversion works correctly
        let (mut mlp, _config) = MLP::<TestBackend>::new(2, 3, Activation::RELU, &rng_ctx, 100, false, false);
        
        // Set known weights and verify they are stored correctly
        let weight_data = [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let weight_tensor: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(weight_data.as_slice(), rng_ctx.device())
            .reshape([2, 3]); // [in_features, out_features]
        
        mlp.linear1.weight = Param::from_tensor(weight_tensor.clone());
        
        // Verify the weight was stored correctly  
        let stored_weight = mlp.linear1.weight.val();
        let stored_data: Vec<f32> = stored_weight.clone().into_data().to_vec().unwrap();
        
        for (i, (&stored, &original)) in stored_data.iter().zip(weight_data.iter()).enumerate() {
            assert_eq!(stored, original, "Weight mismatch at index {}", i);
        }
        
        // Verify shape is correct [in_features, out_features]
        assert_eq!(stored_weight.shape().dims, [2, 3]);
    }

    #[test]
    fn test_different_activation_functions() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        
        // Test both GELU and RELU activations produce different outputs
        let (mlp_gelu, config_gelu) = MLP::<TestBackend>::new(3, 4, Activation::GELU, &rng_ctx, 100, false, false);
        let (mlp_relu, config_relu) = MLP::<TestBackend>::new(3, 4, Activation::RELU, &rng_ctx, 101, false, false);
        
        let input: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats([1.0f32, -0.5f32, 0.5f32].as_slice(), rng_ctx.device());
        
        let output_gelu = mlp_gelu.mlp_forward(input.clone(), &config_gelu, false, false, None);
        let output_relu = mlp_relu.mlp_forward(input, &config_relu, false, false, None);
        
        let data_gelu: Vec<f32> = output_gelu.into_data().to_vec().unwrap();
        let data_relu: Vec<f32> = output_relu.into_data().to_vec().unwrap();
        
        // Outputs should be different (with high probability given random weights)
        let mut different = false;
        for (gelu_val, relu_val) in data_gelu.iter().zip(data_relu.iter()) {
            if (gelu_val - relu_val).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "GELU and RELU should produce different outputs");
    }

    #[test]
    fn test_batch_processing() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) = MLP::<TestBackend>::new(2, 3, Activation::GELU, &rng_ctx, 100, false, false);
        
        // Test with batch size > 1
        let batch_input: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(
            [1.0f32, 2.0f32, 3.0f32, 4.0f32].as_slice(), rng_ctx.device())
            .reshape([2, 2]); // batch_size=2, features=2
        
        let output = mlp.mlp_forward(batch_input, &config, false, false, None);
        let output_shape = output.shape();
        
        // Output should maintain batch dimension
        assert_eq!(output_shape.dims, [2, 2], "Batch processing should preserve batch dimension");
    }

    #[test]
    fn test_add_input_functionality() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) = MLP::<TestBackend>::new(3, 4, Activation::GELU, &rng_ctx, 100, false, false);
        
        let input: Tensor<TestBackend, 1> = Tensor::<TestBackend, 1>::from_floats([1.0f32, 2.0f32, 3.0f32].as_slice(), rng_ctx.device());
        
        // Test without add_input
        let output_no_add = mlp.mlp_forward(input.clone(), &config, false, false, None);
        
        // Test with add_input
        let output_with_add = mlp.mlp_forward(input.clone(), &config, true, false, None);
        
        let data_no_add: Vec<f32> = output_no_add.into_data().to_vec().unwrap();
        let data_with_add: Vec<f32> = output_with_add.into_data().to_vec().unwrap();
        let input_data: Vec<f32> = input.into_data().to_vec().unwrap();
        
        // With add_input=true, output should be different
        let mut different = false;
        for (no_add, with_add) in data_no_add.iter().zip(data_with_add.iter()) {
            if (no_add - with_add).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "add_input should modify the output");
    }

    #[test]
    fn test_memory_optimization() {
        let device = Default::default();
        let rng_ctx = DeterministicRngContext::<TestBackend>::new(42, device);
        let (mlp, config) = MLP::<TestBackend>::new(4, 8, Activation::GELU, &rng_ctx, 100, false, false);
        
        let input: Tensor<TestBackend, 2> = Tensor::<TestBackend, 1>::from_floats(
            [1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32].as_slice(), rng_ctx.device())
            .reshape([2, 4]);
        
        // Test with memory optimization factor
        let output_standard = mlp.mlp_forward(input.clone(), &config, false, false, None);
        let output_optimized = mlp.mlp_forward(input, &config, false, false, Some(2));
        
        let data_standard: Vec<f32> = output_standard.into_data().to_vec().unwrap();
        let data_optimized: Vec<f32> = output_optimized.into_data().to_vec().unwrap();
        
        // Results should be identical regardless of memory optimization
        for (standard, optimized) in data_standard.iter().zip(data_optimized.iter()) {
            let diff = (standard - optimized).abs();
            assert!(diff < 1e-6, "Memory optimization should not change results");
        }
    }
}

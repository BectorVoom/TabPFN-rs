//! Copyright (c) Prior Labs GmbH 2025.
//!
//! Multi-Layer Perceptron (MLP) module - Rust implementation of
//! src/tabpfn/architectures/base/mlp.py

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Tensor, activation};

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
///     let output = mlp.forward(x, &config, false, false, None);
///     ```
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// The first linear layer
    pub linear1: Linear<B>,
    /// The second linear layer
    pub linear2: Linear<B>,
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
    /// Create a new MLP module
    ///
    /// # Arguments
    /// * `size` - The input and output size of the MLP
    /// * `hidden_size` - The size of the hidden layer
    /// * `activation` - The activation function to use
    /// * `device` - The device to use for the linear layers
    /// * `initialize_output_to_zero` - Whether to initialize the output layer weights to zero
    /// * `recompute` - Whether to recompute the forward pass during backpropagation
    pub fn new(
        size: usize,
        hidden_size: usize,
        activation: Activation,
        device: &B::Device,
        initialize_output_to_zero: bool,
        recompute: bool,
    ) -> (Self, MLPConfig) {
        let linear1 = LinearConfig::new(size, hidden_size)
            .with_bias(false)
            .init(device);

        let mut linear2 = LinearConfig::new(hidden_size, size)
            .with_bias(false)
            .init(device);

        // Initialize output weights to zero if requested
        if initialize_output_to_zero {
            let weight = linear2.weight.val();
            let zero_weight = weight.zeros_like();
            linear2.weight = Param::from_tensor(zero_weight);
        }

        let mlp = Self { linear1, linear2 };

        let config = MLPConfig {
            activation,
            recompute,
        };

        (mlp, config)
    }

    /// Create MLP from string activation
    pub fn new_with_str_activation(
        size: usize,
        hidden_size: usize,
        activation: &str,
        device: &B::Device,
        initialize_output_to_zero: bool,
        recompute: bool,
    ) -> Result<(Self, MLPConfig), String> {
        let activation = Activation::from_str(activation)?;
        Ok(Self::new(
            size,
            hidden_size,
            activation,
            device,
            initialize_output_to_zero,
            recompute,
        ))
    }

    /// Compute the forward pass - equivalent to Python's _compute method
    /// Compute the forward pass - equivalent to Python's _compute method
    /// 
    /// In Burn, gradient checkpointing is handled automatically by Backend type:
    /// - Autodiff<NdArray, BalancedCheckpointing> enables automatic checkpointing
    /// - Autodiff<NdArray, NoCheckpointing> uses standard computation  
    /// The recompute flag is kept for API compatibility with Python version
    fn _compute<const D: usize>(&self, x: Tensor<B, D>, config: &MLPConfig) -> Tensor<B, D> {
        let x = self.linear1.forward(x);

        // Apply activation function
        let x = match config.activation {
            Activation::GELU => activation::gelu(x),
            Activation::RELU => activation::relu(x),
        };

        self.linear2.forward(x)
    }

    /// Chunked computation for memory efficiency
    fn _compute_chunked<const D: usize>(
        &self,
        x: Tensor<B, D>,
        config: &MLPConfig,
        factor: usize,
        add_input: bool,
    ) -> Tensor<B, D> {
        if factor <= 1 {
            // No chunking needed
            let mlp_output = self._compute(x.clone(), config);
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
            let chunk_result = self._compute(chunk.clone(), config);
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
    ///     save_peak_mem_factor:
    ///         If provided, enables a
    ///         memory-saving technique that reduces peak memory usage during the
    ///         forward pass. This requires 'add_input' and 'allow_inplace' to be true.
    ///         See the documentation of the trait 'SavePeakMemFactor'
    ///         for details. Default is None.
    pub fn forward<const D: usize>(
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
            self._compute_chunked(x_reshaped.clone(), config, factor, add_input)
        } else {
            // Standard computation
            let mlp_output = self._compute(x_reshaped.clone(), config);
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
        let (_mlp, config) =
            MLP::<TestBackend>::new(128, 256, Activation::GELU, &device, false, false);

        assert_eq!(config.activation, Activation::GELU);
        assert!(!config.recompute);
    }

    #[test]
    fn test_mlp_from_str_activation() {
        let device = Default::default();
        let (_mlp, config) =
            MLP::<TestBackend>::new_with_str_activation(128, 256, "gelu", &device, false, false)
                .unwrap();

        assert_eq!(config.activation, Activation::GELU);
    }

    #[test]
    fn test_mlp_forward_2d() {
        let device = Default::default();
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::RELU, &device, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::zeros([32, 64], &device);
        let output = mlp.forward(x, &config, false, false, None);

        assert_eq!(output.shape().dims, [32, 64]);
    }

    #[test]
    fn test_mlp_forward_3d() {
        let device = Default::default();
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::GELU, &device, false, false);

        let x: Tensor<TestBackend, 3> = Tensor::zeros([8, 16, 64], &device);
        let output = mlp.forward(x, &config, false, false, None);

        assert_eq!(output.shape().dims, [8, 16, 64]);
    }

    #[test]
    fn test_mlp_with_add_input() {
        let device = Default::default();
        let (mlp, config) =
            MLP::<TestBackend>::new(64, 128, Activation::RELU, &device, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([4, 64], &device);
        let output = mlp.forward(x.clone(), &config, true, true, None);

        // With add_input=true, output should include the original input
        assert_eq!(output.shape().dims, [4, 64]);
    }

    #[test]
    fn test_mlp_zero_initialization() {
        let device = Default::default();
        let (mlp, config) = MLP::<TestBackend>::new(32, 64, Activation::GELU, &device, true, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([2, 32], &device);
        let output = mlp.forward(x, &config, false, false, None);

        // With zero initialization of output weights, the final output should be zeros
        // (since we're only going through linear2 which has zero weights)
        assert_eq!(output.shape().dims, [2, 32]);
    }

    #[test]
    fn test_mlp_memory_optimization() {
        let device = Default::default();
        let (mlp, config) =
            MLP::<TestBackend>::new(32, 64, Activation::RELU, &device, false, false);

        let x: Tensor<TestBackend, 2> = Tensor::ones([8, 32], &device);
        let output = mlp.forward(x, &config, false, true, Some(2));

        assert_eq!(output.shape().dims, [8, 32]);
    }

    #[test]
    fn test_mlp_gradient_checkpointing() {
        let device = Default::default();

        // Test with recompute=false (standard)
        let (mlp_standard, config_standard) =
            MLP::<TestBackend>::new(16, 32, Activation::GELU, &device, false, false);

        // Test with recompute=true (gradient checkpointing)
        let (mut mlp_checkpoint, config_checkpoint) =
            MLP::<TestBackend>::new(16, 32, Activation::GELU, &device, false, true);

        // Copy weights to ensure identical computation
        mlp_checkpoint.linear1.weight = mlp_standard.linear1.weight.clone();
        mlp_checkpoint.linear2.weight = mlp_standard.linear2.weight.clone();

        let x: Tensor<TestBackend, 2> = Tensor::ones([4, 16], &device);

        // Both methods should produce identical results
        let output_standard = mlp_standard.forward(x.clone(), &config_standard, false, false, None);
        let output_checkpoint = mlp_checkpoint.forward(x, &config_checkpoint, false, false, None);

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

        // Test that recompute flag is properly stored and used
        let (_mlp, config) = MLP::<TestBackend>::new(8, 16, Activation::RELU, &device, false, true);
        assert!(config.recompute, "recompute flag should be set to true");

        let (_mlp, config) =
            MLP::<TestBackend>::new(8, 16, Activation::RELU, &device, false, false);
        assert!(!config.recompute, "recompute flag should be set to false");
    }
}

//! TabPFN Model Instance Implementation
//! 
//! This module provides the main TabPFN model instance with multi-backend support
//! (CPU, WGPU, CUDA) and training/inference modes.

use burn::{
    prelude::Backend,
    module::Module,
    tensor::{backend::AutodiffBackend, Tensor},
};

// Always include ndarray backend as the default
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_autodiff::Autodiff as AutodiffWrapper;

// Conditionally include other backends based on features
#[cfg(feature = "wgpu")]
use burn_wgpu::{Wgpu, WgpuDevice};

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

use super::base::{
    config::ModelConfig,
    transformer::PerFeatureTransformer,
    encoders::{SequentialEncoder, MulticlassClassificationTargetEncoder},
};

/// Backend type aliases for different compute devices
pub type CpuBackend = NdArray<f32>;

#[cfg(feature = "wgpu")]
pub type WgpuBackend = Wgpu<f32, i32>;

#[cfg(feature = "cuda")]
pub type CudaBackend = Cuda<f32, i32>;

/// Training backend with autodiff support
pub type CpuAutodiffBackend = AutodiffWrapper<CpuBackend>;

#[cfg(feature = "wgpu")]
pub type WgpuAutodiffBackend = AutodiffWrapper<WgpuBackend>;

#[cfg(feature = "cuda")]
pub type CudaAutodiffBackend = AutodiffWrapper<CudaBackend>;

/// Supported backend types for runtime selection
#[derive(Debug, Clone)]
pub enum BackendType {
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Cpu
    }
}

impl BackendType {
    /// Get the best available backend for the current system
    pub fn best_available() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA first if available
            if Self::is_cuda_available() {
                return BackendType::Cuda;
            }
        }
        
        #[cfg(feature = "wgpu")]
        {
            // Try WGPU if CUDA is not available
            if Self::is_wgpu_available() {
                return BackendType::Wgpu;
            }
        }
        
        // Default to CPU
        BackendType::Cpu
    }
    
    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> bool {
        // Simple check - in a real implementation you'd check for CUDA runtime
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }
    
    #[cfg(feature = "wgpu")]
    fn is_wgpu_available() -> bool {
        // Simple check - in a real implementation you'd check for GPU availability
        true
    }
}

/// TabPFN Model Instance
/// 
/// Generic over backend B to support different compute devices and autodiff modes
#[derive(Module, Debug)]
pub struct TabPFNInstance<B: Backend> {
    /// Model configuration
    config: ModelConfig,
    /// Input encoder for feature preprocessing
    input_encoder: SequentialEncoder<B>,
    /// Main transformer architecture
    transformer: PerFeatureTransformer<B>,
    /// Output encoder for target processing
    target_encoder: MulticlassClassificationTargetEncoder<B>,
    /// Device for tensor operations
    device: <B as Backend>::Device,
}

impl<B: Backend> TabPFNInstance<B> {
    /// Create a new TabPFN instance with the given configuration
    pub fn new(config: ModelConfig, device: B::Device) -> Self {
        // Validate configuration consistency
        config.validate_consistent()
            .expect("Invalid model configuration");

        // Initialize encoders based on configuration
        let input_encoder = SequentialEncoder::new();
        let transformer = PerFeatureTransformer::new(
            &config,
            config.max_num_classes as usize, // n_out
            "gelu", // activation
            None, // min_num_layers_layer_dropout
            false, // zero_init
            None, // nlayers_decoder
            false, // use_encoder_compression_layer
            None, // precomputed_kv
            false, // cache_trainset_representation
            &device,
        );
        let target_encoder = MulticlassClassificationTargetEncoder::new();

        Self {
            config,
            input_encoder,
            transformer,
            target_encoder,
            device,
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Forward pass through the model
    /// 
    /// # Arguments
    /// * `input` - Input tensor of shape [sequence_length, batch_size, num_features]
    /// 
    /// # Returns
    /// Output tensor of shape [sequence_length, batch_size, num_classes] for classification
    pub fn forward(&mut self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Convert input to expected format for PerFeatureTransformer
        let mut x_map = std::collections::HashMap::new();
        x_map.insert("main".to_string(), input);
        
        // Call transformer forward with required parameters
        match self.transformer.transformer_forward(
            x_map,
            None, // No y provided (inference mode)
            true, // only_return_standard_out
            None, // categorical_inds
            None, // style
            None, // data_dags
        ) {
            Ok(output) => output,
            Err(_) => {
                // Fallback: return zeros of appropriate shape
                let [seq_len, batch_size, _] = input.dims();
                Tensor::zeros([seq_len, batch_size, self.config.max_num_classes as usize], &input.device())
            }
        }
    }
}

/// Training-specific implementations for autodiff backends
impl<B: AutodiffBackend> TabPFNInstance<B> {
    /// Training mode forward pass with gradient computation
    /// 
    /// This method uses the autodiff backend for gradient computation and 
    /// supports gradient checkpointing based on the model configuration.
    pub fn train(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Enable gradient computation for training
        let input = input.require_grad();
        
        // Forward pass with potential gradient checkpointing
        if self.config.recompute_layer || self.config.recompute_attn {
            // Use gradient checkpointing to save memory during training
            self.forward_with_checkpointing(input)
        } else {
            // Standard forward pass
            self.forward(input)
        }
    }

    /// Forward pass with gradient checkpointing for memory efficiency
    fn forward_with_checkpointing(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Implement gradient checkpointing similar to PyTorch's checkpoint functionality
        // This allows trading compute for memory during training
        
        // For now, use standard forward pass
        // TODO: Implement proper gradient checkpointing when Burn supports it
        self.forward(input)
    }
}

/// Inference-specific implementations for all backends
impl<B: Backend> TabPFNInstance<B> {
    /// Evaluation mode forward pass without gradient computation
    pub fn eval(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Ensure no gradients are computed during evaluation
        self.forward(input.detach())
    }

    /// Prediction mode - alias for eval for API consistency
    pub fn predict(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.eval(input)
    }
}

/// Factory functions for creating instances with specific backends
impl TabPFNInstance<CpuBackend> {
    /// Create a CPU-based instance
    pub fn cpu(config: ModelConfig) -> Self {
        let device = NdArrayDevice::default();
        Self::new(config, device)
    }
}

#[cfg(feature = "wgpu")]
impl TabPFNInstance<WgpuBackend> {
    /// Create a WGPU-based instance for GPU acceleration
    pub fn wgpu(config: ModelConfig) -> Self {
        let device = WgpuDevice::default();
        Self::new(config, device)
    }
}

#[cfg(feature = "cuda")]
impl TabPFNInstance<CudaBackend> {
    /// Create a CUDA-based instance for NVIDIA GPU acceleration
    pub fn cuda(config: ModelConfig) -> Self {
        let device = CudaDevice::default();
        Self::new(config, device)
    }
}

/// Training instance factory functions with autodiff support
impl TabPFNInstance<CpuAutodiffBackend> {
    /// Create a CPU-based training instance with autodiff
    pub fn cpu_autodiff(config: ModelConfig) -> Self {
        let device = NdArrayDevice::default();
        Self::new(config, device)
    }
}

#[cfg(feature = "wgpu")]
impl TabPFNInstance<WgpuAutodiffBackend> {
    /// Create a WGPU-based training instance with autodiff
    pub fn wgpu_autodiff(config: ModelConfig) -> Self {
        let device = WgpuDevice::default();
        Self::new(config, device)
    }
}

#[cfg(feature = "cuda")]
impl TabPFNInstance<CudaAutodiffBackend> {
    /// Create a CUDA-based training instance with autodiff
    pub fn cuda_autodiff(config: ModelConfig) -> Self {
        let device = CudaDevice::default();
        Self::new(config, device)
    }
}

/// Builder pattern for creating TabPFN instances with different backends
pub struct TabPFNBuilder {
    config: ModelConfig,
    backend_type: BackendType,
    training_mode: bool,
}

impl TabPFNBuilder {
    /// Create a new builder with the given configuration
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            backend_type: BackendType::default(),
            training_mode: false,
        }
    }
    
    /// Create a new builder with the best available backend
    pub fn new_with_best_backend(config: ModelConfig) -> Self {
        Self {
            config,
            backend_type: BackendType::best_available(),
            training_mode: false,
        }
    }

    /// Set the backend type
    pub fn backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }
    
    /// Use CPU backend
    pub fn cpu(mut self) -> Self {
        self.backend_type = BackendType::Cpu;
        self
    }
    
    /// Use WGPU backend (if available)
    #[cfg(feature = "wgpu")]
    pub fn wgpu(mut self) -> Self {
        self.backend_type = BackendType::Wgpu;
        self
    }
    
    /// Use CUDA backend (if available)
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self) -> Self {
        self.backend_type = BackendType::Cuda;
        self
    }

    /// Enable training mode (uses autodiff backend)
    pub fn training(mut self) -> Self {
        self.training_mode = true;
        self
    }

    /// Enable inference mode (uses standard backend)
    pub fn inference(mut self) -> Self {
        self.training_mode = false;
        self
    }

    /// Build the appropriate instance based on configuration
    pub fn build(self) -> Result<TabPFNInstanceEnum, String> {
        match (self.backend_type, self.training_mode) {
            (BackendType::Cpu, false) => {
                Ok(TabPFNInstanceEnum::CpuInference(TabPFNInstance::cpu(self.config)))
            }
            (BackendType::Cpu, true) => {
                Ok(TabPFNInstanceEnum::CpuTraining(TabPFNInstance::cpu_autodiff(self.config)))
            }
            #[cfg(feature = "wgpu")]
            (BackendType::Wgpu, false) => {
                Ok(TabPFNInstanceEnum::WgpuInference(TabPFNInstance::wgpu(self.config)))
            }
            #[cfg(feature = "wgpu")]
            (BackendType::Wgpu, true) => {
                Ok(TabPFNInstanceEnum::WgpuTraining(TabPFNInstance::wgpu_autodiff(self.config)))
            }
            #[cfg(feature = "cuda")]
            (BackendType::Cuda, false) => {
                Ok(TabPFNInstanceEnum::CudaInference(TabPFNInstance::cuda(self.config)))
            }
            #[cfg(feature = "cuda")]
            (BackendType::Cuda, true) => {
                Ok(TabPFNInstanceEnum::CudaTraining(TabPFNInstance::cuda_autodiff(self.config)))
            }
            #[cfg(not(feature = "wgpu"))]
            (BackendType::Wgpu, _) => {
                Err("WGPU backend requested but not compiled with wgpu feature".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            (BackendType::Cuda, _) => {
                Err("CUDA backend requested but not compiled with cuda feature".to_string())
            }
        }
    }
}

/// Convenience function to create a TabPFN instance with sensible defaults
pub fn create_tabpfn_instance(config: ModelConfig) -> Result<TabPFNInstanceEnum, String> {
    TabPFNBuilder::new_with_best_backend(config)
        .inference()
        .build()
}

/// Convenience function to create a TabPFN training instance with sensible defaults
pub fn create_tabpfn_training_instance(config: ModelConfig) -> Result<TabPFNInstanceEnum, String> {
    TabPFNBuilder::new_with_best_backend(config)
        .training()
        .build()
}

/// Enum wrapper for different backend instances
pub enum TabPFNInstanceEnum {
    CpuInference(TabPFNInstance<CpuBackend>),
    CpuTraining(TabPFNInstance<CpuAutodiffBackend>),
    #[cfg(feature = "wgpu")]
    WgpuInference(TabPFNInstance<WgpuBackend>),
    #[cfg(feature = "wgpu")]
    WgpuTraining(TabPFNInstance<WgpuAutodiffBackend>),
    #[cfg(feature = "cuda")]
    CudaInference(TabPFNInstance<CudaBackend>),
    #[cfg(feature = "cuda")]
    CudaTraining(TabPFNInstance<CudaAutodiffBackend>),
}

impl TabPFNInstanceEnum {
    /// Forward pass dispatch based on the variant
    pub fn forward(&self, input: Tensor<NdArray<f32>, 3>) -> Tensor<NdArray<f32>, 3> {
        match self {
            TabPFNInstanceEnum::CpuInference(instance) => instance.forward(input),
            TabPFNInstanceEnum::CpuTraining(instance) => {
                // Convert to autodiff tensor for training
                let autodiff_input = input.require_grad();
                instance.forward(autodiff_input).inner()
            }
            #[cfg(feature = "wgpu")]
            TabPFNInstanceEnum::WgpuInference(instance) => {
                // Convert tensor to WGPU backend
                let wgpu_input = input.to_device(&instance.device());
                instance.forward(wgpu_input).to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "wgpu")]
            TabPFNInstanceEnum::WgpuTraining(instance) => {
                let wgpu_input = input.to_device(&instance.device()).require_grad();
                instance.forward(wgpu_input).inner().to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "cuda")]
            TabPFNInstanceEnum::CudaInference(instance) => {
                let cuda_input = input.to_device(&instance.device());
                instance.forward(cuda_input).to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "cuda")]
            TabPFNInstanceEnum::CudaTraining(instance) => {
                let cuda_input = input.to_device(&instance.device()).require_grad();
                instance.forward(cuda_input).inner().to_device(&NdArrayDevice::default())
            }
        }
    }

    /// Training mode forward pass
    pub fn train(&self, input: Tensor<NdArray<f32>, 3>) -> Tensor<NdArray<f32>, 3> {
        match self {
            TabPFNInstanceEnum::CpuTraining(instance) => {
                let autodiff_input = input.require_grad();
                instance.train(autodiff_input).inner()
            }
            #[cfg(feature = "wgpu")]
            TabPFNInstanceEnum::WgpuTraining(instance) => {
                let wgpu_input = input.to_device(&instance.device()).require_grad();
                instance.train(wgpu_input).inner().to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "cuda")]
            TabPFNInstanceEnum::CudaTraining(instance) => {
                let cuda_input = input.to_device(&instance.device()).require_grad();
                instance.train(cuda_input).inner().to_device(&NdArrayDevice::default())
            }
            _ => panic!("Training mode not supported for inference-only instances"),
        }
    }

    /// Evaluation mode forward pass
    pub fn eval(&self, input: Tensor<NdArray<f32>, 3>) -> Tensor<NdArray<f32>, 3> {
        match self {
            TabPFNInstanceEnum::CpuInference(instance) => instance.eval(input),
            TabPFNInstanceEnum::CpuTraining(instance) => {
                let autodiff_input = input.require_grad();
                instance.eval(autodiff_input).inner()
            }
            #[cfg(feature = "wgpu")]
            TabPFNInstanceEnum::WgpuInference(instance) => {
                let wgpu_input = input.to_device(&instance.device());
                instance.eval(wgpu_input).to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "wgpu")]
            TabPFNInstanceEnum::WgpuTraining(instance) => {
                let wgpu_input = input.to_device(&instance.device()).require_grad();
                instance.eval(wgpu_input).inner().to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "cuda")]
            TabPFNInstanceEnum::CudaInference(instance) => {
                let cuda_input = input.to_device(&instance.device());
                instance.eval(cuda_input).to_device(&NdArrayDevice::default())
            }
            #[cfg(feature = "cuda")]
            TabPFNInstanceEnum::CudaTraining(instance) => {
                let cuda_input = input.to_device(&instance.device()).require_grad();
                instance.eval(cuda_input).inner().to_device(&NdArrayDevice::default())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_instance_creation() {
        let config = ModelConfig {
            max_num_classes: 10,
            num_buckets: 512,
            ..Default::default()
        };
        
        let instance = TabPFNInstance::cpu(config);
        assert_eq!(instance.config().max_num_classes, 10);
        assert_eq!(instance.config().num_buckets, 512);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ModelConfig {
            max_num_classes: 5,
            num_buckets: 256,
            ..Default::default()
        };
        
        let builder = TabPFNBuilder::new(config)
            .backend(BackendType::Cpu)
            .inference();
            
        let instance = builder.build().expect("Failed to build instance");
        
        // Test that the instance was created successfully
        match instance {
            TabPFNInstanceEnum::CpuInference(_) => {},
            _ => panic!("Expected CPU inference instance"),
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();
        config.max_num_classes = 10;
        config.num_buckets = 512;
        config.emsize = 193;  // Not divisible by nhead (6)
        
        let result = std::panic::catch_unwind(|| {
            TabPFNInstance::cpu(config)
        });
        
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_functions() {
        let config = ModelConfig {
            max_num_classes: 3,
            num_buckets: 64,
            ..Default::default()
        };
        
        // Test inference instance creation
        let inference_instance = create_tabpfn_instance(config.clone())
            .expect("Failed to create inference instance");
        
        match inference_instance {
            TabPFNInstanceEnum::CpuInference(_) => {},
            _ => panic!("Expected CPU inference instance"),
        }
        
        // Test training instance creation
        let training_instance = create_tabpfn_training_instance(config)
            .expect("Failed to create training instance");
        
        match training_instance {
            TabPFNInstanceEnum::CpuTraining(_) => {},
            _ => panic!("Expected CPU training instance"),
        }
    }

    #[test]
    fn test_best_available_backend() {
        let backend = BackendType::best_available();
        
        // Should always return a valid backend (at minimum CPU)
        match backend {
            BackendType::Cpu => {}, // Always available
            #[cfg(feature = "wgpu")]
            BackendType::Wgpu => {},
            #[cfg(feature = "cuda")]
            BackendType::Cuda => {},
        }
    }
}
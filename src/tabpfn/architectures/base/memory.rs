//! Copyright (c) Prior Labs GmbH 2025.
//!
//! Rust implementation of TabPFN memory utilities - semantically equivalent to
//! src/tabpfn/architectures/base/memory.py

use burn::module::Module;
use burn::prelude::*;
use burn::tensor::Tensor;

use std::env;
use std::sync::Once;

use crate::tabpfn::settings::settings;

// Constants from Python implementation
pub const SAVE_PEAK_MEM_FACTOR: usize = 8;

// TODO(eddiebergman): pulled from `def _estimate_model_usage()`
pub const CONSTANT_MEMORY_OVERHEAD: i64 = 100_000_000;
pub const MEMORY_FACTOR_SAVE_PEAK_MEM_ACTIVE: f64 = 2.5;
pub const DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA: f64 = 8.0;

// TODO(eddiebergman): pulled from `def _estimate_model_usage()`
// Had it's own todo of "check if correct"
pub const NUM_SAMPLES_FACTOR: f64 = 4.0;
pub const NUM_SAMPLES_PLUS_FEATURES: f64 = 6.5;
pub const CELLS_FACTOR: f64 = 0.25;
pub const CELLS_SQUARED_FACTOR: f64 = 1.3e-7;



// Initialize PyTorch CUDA allocation config equivalent
static INIT: Once = Once::new();

pub fn initialize_memory_config() {
    INIT.call_once(|| {
        unsafe {
            env::set_var("PYTORCH_CUDA_ALLOC_CONF", &settings().pytorch.pytorch_cuda_alloc_conf);
        }
    });
}

/// Memory unit types
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryUnit {
    Bytes,
    Megabytes,
    Gigabytes,
}

impl MemoryUnit {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryUnit::Bytes => "b",
            MemoryUnit::Megabytes => "mb",
            MemoryUnit::Gigabytes => "gb",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "b" => Ok(MemoryUnit::Bytes),
            "mb" => Ok(MemoryUnit::Megabytes),
            "gb" => Ok(MemoryUnit::Gigabytes),
            _ => Err(format!("Invalid unit {}. Must be one of 'b', 'mb', or 'gb'", s)),
        }
    }

    pub fn conversion_factor(&self) -> f64 {
        match self {
            MemoryUnit::Bytes => 1.0,
            MemoryUnit::Megabytes => 1e6,
            MemoryUnit::Gigabytes => 1e9,
        }
    }
}

/// Trait for memory-optimized operations on modules
/// 
/// This is the Rust equivalent of the Python decorator `support_save_peak_mem_factor`
pub trait SavePeakMemFactor<B: Backend> {
    /// Apply memory optimization to a method operation
    /// 
    /// Args:
    ///     x: The input tensor
    ///     add_input: Whether to add the input to the result
    ///     allow_inplace: Whether inplace operations are allowed
    ///     save_peak_mem_factor: Number of chunks to split operation into
    ///     operation: The operation to apply
    /// 
    /// Returns:
    ///     The result tensor
    fn apply_with_memory_optimization<const D: usize>(
        &self,
        x: Tensor<B, D>,
        add_input: bool,
        allow_inplace: bool,
        save_peak_mem_factor: Option<usize>,
        operation: impl Fn(&Self, Tensor<B, D>) -> Tensor<B, D>,
    ) -> Tensor<B, D>
    where
        Self: Module<B>;
}

impl<B: Backend, M: Module<B>> SavePeakMemFactor<B> for M {
    fn apply_with_memory_optimization<const D: usize>(
        &self,
        x: Tensor<B, D>,
        add_input: bool,
        allow_inplace: bool,
        save_peak_mem_factor: Option<usize>,
        operation: impl Fn(&Self, Tensor<B, D>) -> Tensor<B, D>,
    ) -> Tensor<B, D> {
        // Validate parameters similar to Python implementation
        if save_peak_mem_factor.is_some() && !allow_inplace {
            panic!("The parameter save_peak_mem_factor only supported with 'allow_inplace' set.");
        }

        if let Some(factor) = save_peak_mem_factor {
            assert!(factor > 1, "save_peak_mem_factor must be greater than 1");

            let batch_size = x.dims()[0];
            let split_size = (batch_size + factor - 1) / factor;

            // Split the tensor along the first dimension
            let device = x.device();
            let mut result = x.clone();
            let splits = x.chunk(factor, 0);

            for (i, chunk) in splits.iter().enumerate() {
                let chunk_result = operation(self, chunk.clone());
                
                if add_input {
                    // In-place addition equivalent to x_[:] += method(...)
                    let start = i * split_size;
                    let end = std::cmp::min(start + split_size, batch_size);
                    let indices = Tensor::arange(start as i64..end as i64, &device);
                    result = result.select_assign(0, indices, chunk.clone() + chunk_result);
                } else {
                    // In-place assignment equivalent to x_[:] = method(...)
                    let start = i * split_size;
                    let end = std::cmp::min(start + split_size, batch_size);
                    let indices = Tensor::arange(start as i64..end as i64, &device);
                    result = result.select_assign(0, indices, chunk_result);
                }
            }
            return result;
        }

        if add_input {
            x.clone() + operation(self, x)
        } else {
            operation(self, x)
        }
    }
}

/// Memory usage estimator - equivalent to Python MemoryUsageEstimator class
pub struct MemoryUsageEstimator;

impl MemoryUsageEstimator {
    pub const SAVE_PEAK_MEM_FACTOR: usize = 8;

    /// Convert a value from one unit to another
    pub fn convert_units(
        value: f64,
        from_unit: MemoryUnit,
        to_unit: MemoryUnit,
    ) -> f64 {
        (value * from_unit.conversion_factor()) / to_unit.conversion_factor()
    }

    /// Convenience method to convert bytes to a different unit
    pub fn convert_bytes_to_unit(value: f64, unit: MemoryUnit) -> f64 {
        Self::convert_units(value, MemoryUnit::Bytes, unit)
    }

    /// Estimate the memory usage of a single batch
    /// 
    /// The calculation is done based on the assumption that save_peak_mem_factor
    /// is not used (since this estimation is used to determine whether to use it).
    /// 
    /// Args:
    ///     x: The input tensor
    ///     ninp: Model embedding size
    ///     features_per_group: Features per group in model
    ///     n_layers: Number of transformer layers
    ///     cache_kv: Whether key and value tensors are cached
    ///     dtype_byte_size: The size of the data type in bytes
    ///     unit: The unit to convert the memory usage to
    ///     n_train_samples: The number of training samples (only for cache_kv mode)
    ///     model_params_count: Number of model parameters
    /// 
    /// Returns:
    ///     The estimated memory usage of a single batch
    pub fn estimate_memory_of_one_batch<B: Backend, const D: usize>(
        x: &Tensor<B, D>,
        ninp: usize,
        features_per_group: usize,
        n_layers: Option<usize>,
        cache_kv: bool,
        dtype_byte_size: usize,
        unit: MemoryUnit,
        n_train_samples: Option<usize>,
        model_params_count: usize,
    ) -> Result<f64, String> {
        let dims = x.dims();
        
        // Validate tensor dimensions - must be 2D or 3D
        if dims.len() < 2 || dims.len() > 3 {
            return Err("X must be a 2D or 3D tensor".to_string());
        }

        if cache_kv && n_train_samples.is_none() {
            return Err("n_train_samples must be provided when cache_kv is True".to_string());
        }

        // Default to 12 layers if not provided (matches Python warning behavior)
        let n_layers = n_layers.unwrap_or_else(|| {
            log::warn!(
                "Could not estimate number of encoder/decoder layers in the \
                transformer model, defaulting to 12."
            );
            12
        });

        let n_samples = dims[dims.len() - 2];
        let n_features = dims[dims.len() - 1];
        let n_batches = if dims.len() == 3 { dims[0] } else { 1 };

        let n_feature_groups = (n_features as f64 / features_per_group as f64).ceil() as usize + 1;

        let model_mem = model_params_count * dtype_byte_size;
        let x_mem = n_samples * n_feature_groups * dtype_byte_size;
        let activation_mem = n_samples * n_feature_groups * ninp * n_layers * dtype_byte_size * n_batches;

        let mut total_mem_bytes = model_mem + x_mem + activation_mem;

        if cache_kv {
            let cached_mem = n_train_samples.unwrap() * n_feature_groups * ninp * 2 * n_layers * dtype_byte_size;
            total_mem_bytes += cached_mem;
        }

        Ok(Self::convert_bytes_to_unit(total_mem_bytes as f64, unit))
    }

    /// Get available free memory for MPS devices
    /// 
    /// Note: This is a placeholder implementation as Metal API access 
    /// would require platform-specific dependencies
    fn _get_mps_free_memory() -> Result<f64, String> {
        // TODO: Implement MPS memory detection for macOS
        // This would require Metal framework bindings
        Err("MPS memory detection not yet implemented in Rust".to_string())
    }

    /// Get maximum free memory available on the system
    /// 
    /// Args:
    ///     device_type: The device type ("cpu", "cuda", "mps")
    ///     unit: The unit to return memory in
    ///     default_gb_cpu_if_failed_to_calculate: Default CPU memory if calculation fails
    /// 
    /// Returns:
    ///     The maximum memory usage in the specified unit
    pub fn get_max_free_memory(
        device_type: &str,
        unit: MemoryUnit,
        default_gb_cpu_if_failed_to_calculate: f64,
    ) -> f64 {
        let free_memory_bytes = match device_type {
            device_type if device_type.starts_with("cpu") => {
                Self::get_cpu_memory().unwrap_or_else(|_| {
                    log::warn!(
                        "Could not get system memory, defaulting to {} GB",
                        default_gb_cpu_if_failed_to_calculate
                    );
                    Self::convert_units(
                        default_gb_cpu_if_failed_to_calculate,
                        MemoryUnit::Gigabytes,
                        MemoryUnit::Bytes,
                    )
                })
            }
            device_type if device_type.starts_with("cuda") => {
                // TODO: Implement CUDA memory detection
                // This would require CUDA runtime bindings
                log::warn!("CUDA memory detection not yet implemented in Rust");
                Self::convert_units(
                    default_gb_cpu_if_failed_to_calculate,
                    MemoryUnit::Gigabytes,
                    MemoryUnit::Bytes,
                )
            }
            device_type if device_type.starts_with("mps") => {
                Self::_get_mps_free_memory().unwrap_or_else(|_| {
                    log::warn!("MPS memory detection failed, using default");
                    Self::convert_units(
                        default_gb_cpu_if_failed_to_calculate,
                        MemoryUnit::Gigabytes,
                        MemoryUnit::Bytes,
                    )
                })
            }
            _ => {
                panic!("Unknown device type: {}", device_type);
            }
        };

        Self::convert_bytes_to_unit(free_memory_bytes, unit)
    }

    /// Get CPU memory using platform-specific methods
    fn get_cpu_memory() -> Result<f64, String> {
        #[cfg(unix)]
        {
            unsafe {
                let sc_page_size = libc::sysconf(libc::_SC_PAGE_SIZE);
                let sc_phys_pages = libc::sysconf(libc::_SC_PHYS_PAGES);
                
                if sc_page_size > 0 && sc_phys_pages > 0 {
                    Ok((sc_page_size * sc_phys_pages) as f64)
                } else {
                    Err("Failed to get system memory via sysconf".to_string())
                }
            }
        }
        #[cfg(windows)]
        {
            // TODO: Implement Windows memory detection
            // This would require Windows API bindings
            Err("Windows memory detection not yet implemented".to_string())
        }
        #[cfg(not(any(unix, windows)))]
        {
            Err("Memory detection not supported on this platform".to_string())
        }
    }

    /// Estimate memory remainder after batch processing
    pub fn estimate_memory_remainder_after_batch<B: Backend, const D: usize>(
        x: &Tensor<B, D>,
        ninp: usize,
        features_per_group: usize,
        n_layers: Option<usize>,
        cache_kv: bool,
        device_type: &str,
        dtype_byte_size: usize,
        safety_factor: f64,
        n_train_samples: Option<usize>,
        model_params_count: usize,
        max_free_mem: Option<f64>,
    ) -> Result<f64, String> {
        let max_free_mem = max_free_mem.unwrap_or_else(|| {
            Self::get_max_free_memory(
                device_type,
                MemoryUnit::Gigabytes,
                DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA,
            )
        });

        let mem_per_batch = Self::estimate_memory_of_one_batch(
            x,
            ninp,
            features_per_group,
            n_layers,
            cache_kv,
            dtype_byte_size,
            MemoryUnit::Gigabytes,
            n_train_samples,
            model_params_count,
        )?;

        Ok(max_free_mem - (mem_per_batch * safety_factor))
    }

    /// Reset peak memory optimization if required
    /// 
    /// Args:
    ///     save_peak_mem: Memory saving configuration
    ///     x: Input tensor
    ///     ninp: Model embedding size  
    ///     features_per_group: Features per group
    ///     n_layers: Number of layers
    ///     cache_kv: Whether to cache key-value pairs
    ///     device_type: Device type string
    ///     dtype_byte_size: Size of data type in bytes
    ///     safety_factor: Safety factor for memory calculation
    ///     n_train_samples: Number of training samples
    ///     model_params_count: Number of model parameters
    ///     reset_callback: Callback to reset the save_peak_mem_factor on the model
    pub fn reset_peak_memory_if_required<B: Backend, const D: usize>(
        save_peak_mem: SavePeakMemConfig,
        x: &Tensor<B, D>,
        ninp: usize,
        features_per_group: usize,
        n_layers: Option<usize>,
        cache_kv: bool,
        device_type: &str,
        dtype_byte_size: usize,
        safety_factor: f64,
        n_train_samples: Option<usize>,
        model_params_count: usize,
        reset_callback: impl Fn(Option<usize>),
    ) -> Result<(), String> {
        let should_save_peak_mem = match save_peak_mem {
            SavePeakMemConfig::Bool(value) => value,
            SavePeakMemConfig::Auto => {
                let memory_available = Self::estimate_memory_remainder_after_batch(
                    x,
                    ninp,
                    features_per_group,
                    n_layers,
                    cache_kv,
                    device_type,
                    dtype_byte_size,
                    safety_factor,
                    n_train_samples,
                    model_params_count,
                    None,
                )?;
                memory_available < 0.0
            }
            SavePeakMemConfig::MaxMemory(max_mem) => {
                let memory_available = Self::estimate_memory_remainder_after_batch(
                    x,
                    ninp,
                    features_per_group,
                    n_layers,
                    cache_kv,
                    device_type,
                    dtype_byte_size,
                    safety_factor,
                    n_train_samples,
                    model_params_count,
                    Some(max_mem),
                )?;
                memory_available < 0.0
            }
        };

        if should_save_peak_mem {
            reset_callback(Some(Self::SAVE_PEAK_MEM_FACTOR));
        } else {
            reset_callback(None);
        }

        Ok(())
    }
}

/// Configuration for save peak memory behavior
#[derive(Debug, Clone)]
pub enum SavePeakMemConfig {
    Bool(bool),
    Auto,
    MaxMemory(f64),
}

impl SavePeakMemConfig {
    pub fn from_value(value: serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::Bool(b) => Ok(SavePeakMemConfig::Bool(b)),
            serde_json::Value::String(s) if s == "auto" => Ok(SavePeakMemConfig::Auto),
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    Ok(SavePeakMemConfig::MaxMemory(f))
                } else {
                    Err("Invalid number for save_peak_mem".to_string())
                }
            }
            _ => Err("save_peak_mem must be bool, 'auto', or number".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_memory_unit_conversion() {
        assert_eq!(MemoryUnit::Bytes.conversion_factor(), 1.0);
        assert_eq!(MemoryUnit::Megabytes.conversion_factor(), 1e6);
        assert_eq!(MemoryUnit::Gigabytes.conversion_factor(), 1e9);
    }

    #[test]
    fn test_memory_unit_from_str() {
        assert_eq!(MemoryUnit::from_str("b").unwrap(), MemoryUnit::Bytes);
        assert_eq!(MemoryUnit::from_str("mb").unwrap(), MemoryUnit::Megabytes);
        assert_eq!(MemoryUnit::from_str("gb").unwrap(), MemoryUnit::Gigabytes);
        assert!(MemoryUnit::from_str("invalid").is_err());
    }

    #[test]
    fn test_convert_units() {
        let result = MemoryUsageEstimator::convert_units(
            1.0,
            MemoryUnit::Gigabytes,
            MemoryUnit::Megabytes,
        );
        assert_eq!(result, 1000.0);
    }

    #[test]
    fn test_convert_bytes_to_unit() {
        let result = MemoryUsageEstimator::convert_bytes_to_unit(1e9, MemoryUnit::Gigabytes);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_save_peak_mem_config_from_value() {
        use serde_json::json;

        let bool_config = SavePeakMemConfig::from_value(json!(true)).unwrap();
        matches!(bool_config, SavePeakMemConfig::Bool(true));

        let auto_config = SavePeakMemConfig::from_value(json!("auto")).unwrap();
        matches!(auto_config, SavePeakMemConfig::Auto);

        let memory_config = SavePeakMemConfig::from_value(json!(8.0)).unwrap();
        matches!(memory_config, SavePeakMemConfig::MaxMemory(8.0));
    }

    #[test]
    fn test_estimate_memory_of_one_batch() {
        let device = Default::default();
        let x: Tensor<TestBackend, 2> = Tensor::zeros([100, 50], &device);
        
        let result = MemoryUsageEstimator::estimate_memory_of_one_batch(
            &x,
            128,  // ninp
            10,   // features_per_group
            Some(12), // n_layers
            false, // cache_kv
            4,    // dtype_byte_size (float32)
            MemoryUnit::Bytes,
            None, // n_train_samples
            1000000, // model_params_count
        );
        
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_estimate_memory_invalid_tensor_dims() {
        let device = Default::default();
        let x: Tensor<TestBackend, 1> = Tensor::zeros([100], &device);
        
        let result = MemoryUsageEstimator::estimate_memory_of_one_batch(
            &x,
            128,
            10,
            Some(12),
            false,
            4,
            MemoryUnit::Bytes,
            None,
            1000000,
        );
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("2D or 3D tensor"));
    }

    #[test]
    fn test_cache_kv_requires_n_train_samples() {
        let device = Default::default();
        let x: Tensor<TestBackend, 2> = Tensor::zeros([100, 50], &device);
        
        let result = MemoryUsageEstimator::estimate_memory_of_one_batch(
            &x,
            128,
            10,
            Some(12),
            true, // cache_kv = true
            4,
            MemoryUnit::Bytes,
            None, // n_train_samples = None
            1000000,
        );
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("n_train_samples must be provided"));
    }
}
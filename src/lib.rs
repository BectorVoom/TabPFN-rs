//! TabPFN-rs: Rust implementation of TabPFN (foundation model for tabular data)

pub mod tabpfn;
// pub mod training;

pub use tabpfn::settings::{Settings, settings};

/// Tensor slice assignment functionality
pub mod tensor_slice_assign;

/// Test utilities for backend-aware tensor construction
/// 
/// Provides helper functions for creating tensors that work with Burn 0.18's
/// Into<TensorData> trait bounds using Vec<T> + .as_slice() pattern.
pub mod test_utils;
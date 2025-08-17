//! TabPFN-rs: Rust implementation of TabPFN (foundation model for tabular data)

pub mod tabpfn;

pub use tabpfn::settings::{Settings, settings};

/// Tensor slice assignment functionality
pub mod tensor_slice_assign;
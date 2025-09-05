# docs.rs Publishing Instructions

This document provides instructions for publishing TabPFN-rs documentation to docs.rs.

## Prerequisites

- Crate must be published to crates.io first
- Documentation must build successfully with `cargo doc`
- All dependencies must be available on crates.io

## Current Crate Information

- **Crate Name:** `tab_pfn_rs`
- **Version:** `0.1.0` (as specified in `Cargo.toml`)
- **Edition:** `2024`
- **Target Rust Version:** `1.83.0`

## Step 1: Verify Documentation Builds

Before publishing, ensure documentation builds correctly:

```bash
# Generate documentation locally (recommended for maintainers)
cargo doc --no-deps

# Generate documentation with all features (if features are defined)
cargo doc --no-deps --all-features

# Open documentation in browser for review
cargo doc --no-deps --open

# Generate documentation with verbose output for debugging
cargo doc --no-deps -v
```

**Expected Output:**
```
   Documenting tab_pfn_rs v0.1.0 (/Users/ods/Documents/TabPFN-rs-main)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.23s
     Generated /Users/ods/Documents/TabPFN-rs-main/target/doc/tab_pfn_rs/index.html
```

**Note for Maintainers:** The `--no-deps` flag is crucial to avoid building documentation for dependencies, which speeds up the process and focuses on the crate's own documentation. Always verify documentation builds successfully before publishing.

## Step 2: Prepare for crates.io Publishing

### 2.1 Update Cargo.toml Metadata

Ensure `Cargo.toml` contains proper metadata for crates.io:

```toml
[package]
name = "tab_pfn_rs"
version = "0.1.0"
edition = "2024"
authors = ["Your Name <email@example.com>"]
description = "Rust implementation of TabPFN (foundation model for tabular data)"
repository = "https://github.com/your-org/TabPFN-rs"
license = "MIT OR Apache-2.0"
keywords = ["machine-learning", "tabular", "transformer", "foundation-model"]
categories = ["science", "algorithms"]
readme = "README.md"

[package.metadata.docs.rs]
# Enable all features for docs.rs
all-features = true
# Use specific Rust version if needed
# rustc-args = ["--cfg", "docsrs"]
```

### 2.2 Create Documentation Examples

Add documentation examples to key public APIs:

```rust
//! # TabPFN-rs: Rust implementation of TabPFN
//! 
//! This crate provides a Rust implementation of TabPFN (Tabular Prior-Data Fitted Networks),
//! a foundation model for tabular data classification tasks.
//! 
//! ## Quick Start
//! 
//! ```rust,no_run
//! use tab_pfn_rs::tabpfn::{TabPFNTrainer, TrainingConfig};
//! // ... example usage
//! ```

impl TabPFNTrainer {
    /// Creates a new TabPFN trainer
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// # use tab_pfn_rs::*;
    /// let config = TrainingConfig::default();
    /// // let trainer = TabPFNTrainer::new(config, &device, rng_context);
    /// ```
    pub fn new(/* ... */) {
        // ...
    }
}
```

## Step 3: Publish to crates.io

### 3.1 Login to crates.io

```bash
# Login with your crates.io token
cargo login

# Or set the token directly
export CARGO_REGISTRY_TOKEN="your-token-here"
```

### 3.2 Dry Run Publication

```bash
# Perform a dry run to check for issues
cargo publish --dry-run

# Check what files will be included
cargo package --list
```

### 3.3 Actual Publication

```bash
# Publish to crates.io
cargo publish
```

## Step 4: Trigger docs.rs Build

After successful publication to crates.io:

1. **Automatic Build:** docs.rs will automatically detect the new crate and build documentation
2. **Manual Trigger:** If needed, you can trigger builds via the docs.rs interface
3. **Build Status:** Check https://docs.rs/crate/tab_pfn_rs/0.1.0/builds for build status

## Step 5: Verify Documentation

1. **Check Build Status:** Visit https://docs.rs/tab_pfn_rs/0.1.0/ 
2. **Verify Content:** Ensure all modules and functions are properly documented
3. **Test Links:** Verify internal and external links work correctly

## Troubleshooting

### Common Issues

1. **Missing Dependencies:**
   ```
   Error: failed to verify package dependencies
   ```
   **Solution:** Ensure all dependencies in `Cargo.toml` are available on crates.io

2. **Documentation Build Failures:**
   ```
   Error: documentation generation failed
   ```
   **Solution:** Run `cargo doc --no-deps` locally and fix any warnings/errors

3. **Feature Flag Issues:**
   ```
   Error: failed to build with feature X
   ```
   **Solution:** Test with `cargo doc --all-features` locally

### Advanced Configuration

For complex documentation needs, add to `Cargo.toml`:

```toml
[package.metadata.docs.rs]
# Build with specific features
features = ["wgpu", "cuda"]

# Set rustdoc arguments
rustdoc-args = ["--cfg", "docsrs"]

# Specify targets for cross-platform docs
targets = ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"]
```

## Publishing Checklist

- [ ] `cargo doc --no-deps` builds successfully
- [ ] `cargo doc --all-features` builds successfully (or `cargo doc --no-deps` if no features defined)
- [ ] All public APIs have documentation
- [ ] Examples in documentation compile
- [ ] `Cargo.toml` metadata is complete
- [ ] `cargo publish --dry-run` succeeds
- [ ] Published to crates.io successfully
- [ ] docs.rs build completes successfully
- [ ] Documentation is accessible at docs.rs URL

## Maintainer Instructions

### Quick Documentation Build for Development

For maintainers working on TabPFN-rs implementation:

```bash
# Quick documentation build for current implementation
cargo doc --no-deps

# View the generated documentation
cargo doc --no-deps --open

# Check for documentation warnings
cargo doc --no-deps 2>&1 | grep warning
```

### Current Implementation Documentation Focus

The current TabPFN-rs implementation includes:

1. **Core Modules:**
   - `tabpfn::architectures::base::loss_utils` - Safe masked cross-entropy loss
   - `tabpfn::architectures::base::train` - Trainer with gradient accumulation
   - `tabpfn::tabpfn` - Main TabPFN interface

2. **Key Public APIs to Document:**
   - `compute_masked_cross_entropy_loss` function with explicit boolean mask
   - `TabPFNTrainer::train_step` method with scalar accumulation
   - Public trainer configuration structures

3. **Documentation Examples:**
   ```rust
   //! # TabPFN-rs: Rust implementation of TabPFN
   //! 
   //! ```rust,no_run
   //! use tab_pfn_rs::tabpfn::architectures::base::loss_utils;
   //! use burn::tensor::{Tensor, TensorData, Bool, Int};
   //! use burn_ndarray::NdArray;
   //! 
   //! // Example of masked cross-entropy loss
   //! let logits = Tensor::<NdArray<f32>, 2>::zeros([4, 3], &device);
   //! let targets = Tensor::<NdArray<f32>, 1, Int>::zeros([4], &device);
   //! let mask = Tensor::<NdArray<f32>, 1, Bool>::ones([4], &device);
   //! 
   //! let loss = loss_utils::compute_masked_cross_entropy_loss(
   //!     logits, targets, mask, &device
   //! );
   //! ```
   ```

### Testing Documentation Examples

```bash
# Test that documentation examples compile
cargo test --doc

# Build documentation with additional warnings
cargo doc --no-deps -- -W missing-docs -W broken-intra-doc-links
```

## Maintenance

### Updating Documentation

For future releases:

1. Update version in `Cargo.toml`
2. Run documentation build tests
3. Publish new version: `cargo publish`
4. docs.rs will automatically build documentation for the new version

### Documentation Links

Once published, documentation will be available at:
- **Latest Version:** https://docs.rs/tab_pfn_rs/latest/
- **Specific Version:** https://docs.rs/tab_pfn_rs/0.1.0/
- **All Versions:** https://docs.rs/tab_pfn_rs/

## Notes

- docs.rs builds documentation in a sandboxed environment
- Network access is limited during builds
- Build resources (CPU/memory) have limits
- Documentation is rebuilt for each new version published to crates.io
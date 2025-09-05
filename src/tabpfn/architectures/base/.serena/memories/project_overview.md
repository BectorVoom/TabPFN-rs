# TabPFN-rs Project Overview

## Purpose
TabPFN-rs is a Rust port of TabPFN (Table Prior-data Fitted Network), a foundation model for tabular data that outperforms traditional methods while being dramatically faster. The project aims to achieve semantic equivalence between the Python PyTorch implementation and a Rust implementation using the Burn deep learning framework.

## Project Structure
- **Python Implementation**: Located in `TabPFN/` directory - contains the original PyTorch-based implementation
- **Rust Implementation**: Located in `src/` directory - the Rust port using Burn framework
- **Key Component**: The transformer architecture at `src/tabpfn/architectures/base/transformer.rs` is a critical component being ported

## Architecture
The codebase implements a per-feature transformer model that:
- Processes a token per feature and sample
- Extends standard Transformer to operate on a per-feature basis
- Allows processing each feature separately while leveraging self-attention
- Includes encoders, decoders, and optional components like feature positional embedding

## Current Status
- Basic Rust structure exists but with incomplete/simplified implementations
- Many tensor operations need proper Burn equivalents
- Graph operations partially implemented
- Forward method exists but requires completion and testing for equivalence
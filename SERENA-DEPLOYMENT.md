# TabPFN-rs Serena MCP Server Deployment Guide

This document provides comprehensive instructions for building, packaging, and deploying TabPFN-rs on Serena MCP server infrastructure.

## Overview

TabPFN-rs is designed to run efficiently on Serena MCP server with optimized Rust binaries, deterministic RNG capabilities, and optional Python interoperability validation.

## Quick Start

### 1. Build Artifacts

```bash
# Build production artifacts for Serena deployment
./scripts/build-serena-artifacts.sh --with-python

# For Linux target specifically (if building on different platform)
./scripts/build-serena-artifacts.sh --target x86_64-unknown-linux-gnu --with-python
```

### 2. Deploy to Serena

```bash
# Upload the generated package
scp tab_pfn_rs-v*-serena-*.tar.gz serena:/path/to/deployment/

# Extract and setup on Serena
ssh serena
cd /path/to/deployment
tar -xzf tab_pfn_rs-v*-serena-*.tar.gz
cd serena-artifacts
./config/setup-environment.sh
```

### 3. Validate Deployment

```bash
# Test basic functionality
./binaries/tab_pfn_rs --version

# Run Python interop validation (if included)
cd validation
uv run python python_cross_check.py
```

## Detailed Build Process

### Build Script Options

The `build-serena-artifacts.sh` script supports several configuration options:

```bash
./scripts/build-serena-artifacts.sh [OPTIONS]

Options:
  --target TARGET      Build for specific target (e.g., x86_64-unknown-linux-gnu)
  --release-only       Skip debug builds and documentation (faster)
  --with-python        Include Python interop validation files
  --help              Show help message
```

### Build Targets

Common build targets for Serena deployment:

- **Linux x86_64:** `x86_64-unknown-linux-gnu` (most common)
- **Linux ARM64:** `aarch64-unknown-linux-gnu` 
- **Linux musl:** `x86_64-unknown-linux-musl` (static linking)

Example for Linux deployment:
```bash
./scripts/build-serena-artifacts.sh --target x86_64-unknown-linux-gnu --with-python
```

## Artifact Structure

The build process creates a `serena-artifacts/` directory with the following structure:

```
serena-artifacts/
├── binaries/           # Optimized release binaries
│   ├── tab_pfn_rs     # Main executable
│   └── test_*         # Test binaries (for validation)
├── docs/              # API documentation (if built)
│   └── api-documentation.tar.gz
├── config/            # Serena-specific configuration
│   ├── serena-manifest.json       # Deployment manifest
│   ├── setup-environment.sh       # Environment setup script
│   └── DEPLOYMENT-README.md       # Quick deployment guide
├── validation/        # Testing and validation files
│   ├── python_cross_check.py     # Python interop test
│   └── python-setup.md           # Python setup instructions
├── SHA256SUMS         # File checksums for verification
└── MANIFEST.txt       # Complete package manifest
```

## System Requirements

### Minimum Requirements

- **CPU:** x86_64 or ARM64 compatible processor
- **Memory:** 2GB RAM (4GB recommended for large datasets)
- **Storage:** 1GB available disk space
- **OS:** Linux (Ubuntu 18.04+ or equivalent)

### Recommended Setup

- **CPU:** Multi-core processor (4+ cores recommended)
- **Memory:** 8GB RAM for optimal performance
- **Storage:** SSD with 5GB+ available space
- **Network:** Stable connection for dependency downloads

## Serena-Specific Configuration

### Environment Variables

The deployment automatically sets up these environment variables:

```bash
export RUST_BACKTRACE=1      # Enable backtraces for debugging
export RUST_LOG=info         # Set logging level
```

### Directory Structure

The setup script creates these directories:

```
deployment-root/
├── tab_pfn_rs           # Main executable
├── data/
│   ├── input/          # Input data files
│   ├── output/         # Generated outputs
│   └── cache/          # Temporary cache files
└── logs/               # Application logs
```

## Python Interoperability

### Setup Python Environment

If deploying with Python validation:

```bash
# Install Python dependencies on Serena
uv add torch numpy scipy scikit-learn

# Verify installation
uv run python -c "import torch, numpy; print('Python deps OK')"
```

### Running Cross-Language Tests

```bash
cd validation
uv run python python_cross_check.py
```

Expected output:
```
=== Cross-Language Validation ===
✅ RNG determinism test passed
✅ Embedding simulation test passed
✅ All cross-language validation tests passed!
```

## Performance Optimization

### Binary Optimization

The build process includes several optimizations:

- **Release mode:** Full compiler optimizations enabled
- **Link-time optimization:** Reduces binary size and improves performance
- **Target CPU:** Optimized for deployment target architecture
- **Dependency pruning:** Only necessary dependencies included

### Runtime Configuration

For optimal performance on Serena:

```bash
# Set thread count based on available cores
export RAYON_NUM_THREADS=4

# Optimize memory allocation
export MALLOC_CONF="background_thread:true,dirty_decay_ms:5000"
```

## Monitoring and Debugging

### Health Checks

Create a simple health check script:

```bash
#!/bin/bash
# health-check.sh
./binaries/tab_pfn_rs --version > /dev/null
echo "Status: $([[ $? -eq 0 ]] && echo 'OK' || echo 'FAILED')"
```

### Log Configuration

Logs are written to the `logs/` directory with these levels:
- `ERROR`: Critical errors requiring attention
- `WARN`: Warnings that don't stop execution
- `INFO`: General operational information
- `DEBUG`: Detailed debugging information

### Common Issues

1. **Missing Dependencies:**
   ```bash
   # Check library dependencies
   ldd ./binaries/tab_pfn_rs
   ```

2. **Permission Issues:**
   ```bash
   # Fix executable permissions
   chmod +x ./binaries/tab_pfn_rs
   ```

3. **Memory Issues:**
   ```bash
   # Monitor memory usage
   top -p $(pgrep tab_pfn_rs)
   ```

## Security Considerations

### File Permissions

The deployment sets appropriate permissions:
- Executables: `755` (read/write/execute for owner, read/execute for others)
- Configuration: `644` (read/write for owner, read for others)
- Data directories: `755` with appropriate ownership

### Network Security

TabPFN-rs operates locally and doesn't require network access during inference, minimizing security surface area.

### Data Privacy

- All computations are performed locally on Serena
- No external API calls or data transmission
- Deterministic RNG ensures reproducible results without external entropy

## Maintenance

### Updates

To update to a new version:

1. Build new artifacts with the updated codebase
2. Upload and extract the new package
3. Run the setup script to update configuration
4. Validate the deployment

### Backup

Important files to backup:
- Configuration files in `config/`
- Any custom data in `data/`
- Application logs in `logs/`

### Monitoring

Set up monitoring for:
- Process health (`tab_pfn_rs` running status)
- Memory usage
- Disk space usage
- Log file sizes

## Troubleshooting

### Build Issues

**Problem:** Cross-compilation fails
**Solution:** Install target toolchain: `rustup target add x86_64-unknown-linux-gnu`

**Problem:** Link errors during build
**Solution:** Ensure all system dependencies are installed

### Runtime Issues

**Problem:** Binary won't start
**Solution:** Check permissions and library dependencies with `ldd`

**Problem:** Out of memory errors
**Solution:** Increase available RAM or reduce dataset size

**Problem:** Python validation fails
**Solution:** Verify Python dependencies: `uv run python -c "import torch, numpy"`

## Support

For deployment issues:

1. Check the `logs/` directory for error messages
2. Verify all dependencies are installed correctly
3. Ensure adequate system resources are available
4. Refer to the main TabPFN-rs documentation
5. Create an issue in the project repository with deployment details

## Version Information

This deployment guide is for TabPFN-rs v0.1.0 and may need updates for future versions. Always refer to the version-specific documentation included in each artifact package.
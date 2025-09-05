#!/bin/bash
# Serena Artifact Building and Deployment Script for TabPFN-rs
# 
# This script builds production artifacts for deployment on Serena MCP server.
# It creates optimized binaries, documentation, and deployment packages.
#
# Usage:
#   ./scripts/build-serena-artifacts.sh [--target TARGET] [--release-only] [--with-python]
#
# Options:
#   --target TARGET    Specify build target (e.g., x86_64-unknown-linux-gnu)
#   --release-only     Skip debug builds and documentation
#   --with-python      Include Python interop validation files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
BUILD_TARGET=""
RELEASE_ONLY=false
WITH_PYTHON=false
ARTIFACTS_DIR="serena-artifacts"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --release-only)
            RELEASE_ONLY=true
            shift
            ;;
        --with-python)
            WITH_PYTHON=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--target TARGET] [--release-only] [--with-python]"
            echo "  --target TARGET    Specify build target (e.g., x86_64-unknown-linux-gnu)"
            echo "  --release-only     Skip debug builds and documentation"
            echo "  --with-python      Include Python interop validation files"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 TabPFN-rs Serena Artifact Builder${NC}"
echo "====================================="

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Extract crate information
CRATE_NAME=$(grep '^name = ' Cargo.toml | head -1 | sed 's/name = "\([^"]*\)"/\1/')
CRATE_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\([^"]*\)"/\1/')
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${BLUE}📦 Building artifacts for: ${CRATE_NAME} v${CRATE_VERSION}${NC}"
echo -e "${BLUE}📅 Build date: ${BUILD_DATE}${NC}"
if [[ -n "$BUILD_TARGET" ]]; then
    echo -e "${BLUE}🎯 Target platform: ${BUILD_TARGET}${NC}"
fi

# Clean and create artifacts directory
echo -e "\n${YELLOW}🧹 Preparing artifacts directory...${NC}"
rm -rf "$ARTIFACTS_DIR"
mkdir -p "$ARTIFACTS_DIR"/{binaries,docs,config,validation}

# Step 1: Run tests to ensure quality
echo -e "\n${YELLOW}🧪 Step 1: Running test suite...${NC}"
if cargo test --release; then
    echo -e "${GREEN}✅ All tests passed${NC}"
    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ"): All tests passed" > "$ARTIFACTS_DIR/test-results.txt"
else
    echo -e "${RED}❌ Tests failed. Cannot create artifacts with failing tests.${NC}"
    exit 1
fi

# Step 2: Build optimized release binaries
echo -e "\n${YELLOW}🔨 Step 2: Building release binaries...${NC}"

if [[ -n "$BUILD_TARGET" ]]; then
    echo "  - Building for target: $BUILD_TARGET"
    rustup target add "$BUILD_TARGET" || echo "Target may already be installed"
    CARGO_BUILD_CMD="cargo build --release --target $BUILD_TARGET"
    BINARY_PATH="target/$BUILD_TARGET/release"
else
    echo "  - Building for host platform"
    CARGO_BUILD_CMD="cargo build --release"
    BINARY_PATH="target/release"
fi

if $CARGO_BUILD_CMD; then
    echo -e "${GREEN}✅ Release build successful${NC}"
else
    echo -e "${RED}❌ Release build failed${NC}"
    exit 1
fi

# Copy binaries to artifacts directory
echo "  - Copying binaries to artifacts directory..."
find "$BINARY_PATH" -maxdepth 1 -type f -executable -name "*${CRATE_NAME}*" | while read binary; do
    cp "$binary" "$ARTIFACTS_DIR/binaries/"
    echo "    Copied: $(basename "$binary")"
done

# Also copy any test binaries that might be useful
find "$BINARY_PATH" -maxdepth 1 -type f -executable -name "test_*" | head -5 | while read binary; do
    cp "$binary" "$ARTIFACTS_DIR/binaries/"
    echo "    Copied test binary: $(basename "$binary")"
done

# Step 3: Generate documentation (unless release-only)
if [[ "$RELEASE_ONLY" == "false" ]]; then
    echo -e "\n${YELLOW}📚 Step 3: Generating documentation...${NC}"
    
    if cargo doc --no-deps --release; then
        echo -e "${GREEN}✅ Documentation generated${NC}"
        
        # Package documentation
        echo "  - Packaging documentation..."
        if [[ -d "target/doc" ]]; then
            tar -czf "$ARTIFACTS_DIR/docs/api-documentation.tar.gz" -C target doc/
            echo "    Documentation packaged as api-documentation.tar.gz"
        fi
    else
        echo -e "${YELLOW}⚠️  Documentation build failed, continuing without docs${NC}"
    fi
else
    echo -e "\n${YELLOW}📚 Step 3: Skipping documentation (release-only mode)${NC}"
fi

# Step 4: Create configuration files for Serena
echo -e "\n${YELLOW}⚙️  Step 4: Creating Serena configuration files...${NC}"

# Create Serena deployment manifest
cat > "$ARTIFACTS_DIR/config/serena-manifest.json" << EOF
{
    "name": "$CRATE_NAME",
    "version": "$CRATE_VERSION",
    "type": "rust-binary",
    "build_date": "$BUILD_DATE",
    "description": "Rust implementation of TabPFN (foundation model for tabular data)",
    "binaries": [
        {
            "name": "$CRATE_NAME",
            "type": "main-executable",
            "entry_point": true
        }
    ],
    "requirements": {
        "rust_version": "1.83.0",
        "system": {
            "memory_mb": 2048,
            "disk_mb": 1024
        }
    },
    "features": [
        "tabular-data-processing",
        "transformer-architecture",
        "deterministic-rng",
        "cross-platform"
    ]
}
EOF

# Create environment setup script
cat > "$ARTIFACTS_DIR/config/setup-environment.sh" << 'EOF'
#!/bin/bash
# Serena Environment Setup for TabPFN-rs
# This script prepares the Serena environment for running TabPFN-rs

echo "🔧 Setting up TabPFN-rs environment on Serena..."

# Set required environment variables
export RUST_BACKTRACE=1
export RUST_LOG=info

# Create data directories if needed
mkdir -p data/{input,output,cache}
mkdir -p logs

# Set appropriate permissions
chmod +x tab_pfn_rs

echo "✅ Environment setup complete"
echo "To run TabPFN-rs: ./tab_pfn_rs"
EOF

chmod +x "$ARTIFACTS_DIR/config/setup-environment.sh"

# Create deployment README
cat > "$ARTIFACTS_DIR/config/DEPLOYMENT-README.md" << EOF
# TabPFN-rs Serena Deployment Guide

## Overview
This package contains production artifacts for TabPFN-rs v$CRATE_VERSION, optimized for deployment on Serena MCP server.

## Contents
- \`binaries/\` - Optimized release binaries
- \`docs/\` - API documentation (if included)
- \`config/\` - Configuration files and setup scripts
- \`validation/\` - Validation and testing files

## Quick Deployment

1. **Upload artifacts:**
   \`\`\`bash
   # Upload the entire serena-artifacts directory to Serena
   scp -r serena-artifacts/ serena:/path/to/deployment/
   \`\`\`

2. **Setup environment:**
   \`\`\`bash
   cd /path/to/deployment/serena-artifacts
   ./config/setup-environment.sh
   \`\`\`

3. **Run validation:**
   \`\`\`bash
   # Test the deployment
   ./binaries/tab_pfn_rs --version
   \`\`\`

## System Requirements
- Memory: 2GB RAM minimum
- Disk: 1GB available space
- Platform: $(uname -m) (or specified target)

## Build Information
- **Version:** $CRATE_VERSION
- **Build Date:** $BUILD_DATE
- **Rust Version:** $(rustc --version)

## Support
For deployment issues, refer to the main TabPFN-rs documentation or create an issue in the repository.
EOF

# Step 5: Include Python validation files (if requested)
if [[ "$WITH_PYTHON" == "true" ]]; then
    echo -e "\n${YELLOW}🐍 Step 5: Including Python interop files...${NC}"
    
    # Copy Python validation scripts
    if [[ -f "python_cross_check.py" ]]; then
        cp python_cross_check.py "$ARTIFACTS_DIR/validation/"
        echo "  - Added: python_cross_check.py"
    fi
    
    if [[ -f "python_reference.py" ]]; then
        cp python_reference.py "$ARTIFACTS_DIR/validation/"
        echo "  - Added: python_reference.py"
    fi
    
    # Create Python setup instructions
    cat > "$ARTIFACTS_DIR/validation/python-setup.md" << EOF
# Python Validation Setup

## Installation
\`\`\`bash
# On Serena, install Python dependencies:
uv add torch numpy scipy scikit-learn
\`\`\`

## Running Validation
\`\`\`bash
# Test Python-Rust compatibility:
uv run python python_cross_check.py
\`\`\`

## Expected Output
The validation should show "All cross-language validation tests passed!" if the Rust implementation is compatible.
EOF
else
    echo -e "\n${YELLOW}🐍 Step 5: Skipping Python files (not requested)${NC}"
fi

# Step 6: Create checksums and manifest
echo -e "\n${YELLOW}🔒 Step 6: Creating checksums and manifest...${NC}"

# Generate checksums for all files
find "$ARTIFACTS_DIR" -type f -name "*.sh" -o -name "*.json" -o -name "*.md" -o -name "tab_pfn_rs*" -o -name "test_*" -o -name "*.tar.gz" | while read file; do
    shasum -a 256 "$file" >> "$ARTIFACTS_DIR/SHA256SUMS"
done

# Create final manifest
cat > "$ARTIFACTS_DIR/MANIFEST.txt" << EOF
TabPFN-rs Serena Deployment Artifacts
=====================================

Package: $CRATE_NAME v$CRATE_VERSION
Build Date: $BUILD_DATE
Build Host: $(uname -a)

Contents:
$(find "$ARTIFACTS_DIR" -type f | sort | sed 's/^/  /')

File Count: $(find "$ARTIFACTS_DIR" -type f | wc -l) files
Total Size: $(du -sh "$ARTIFACTS_DIR" | cut -f1)

Build Configuration:
- Target: ${BUILD_TARGET:-"host platform"}
- Release Only: $RELEASE_ONLY
- With Python: $WITH_PYTHON

Verification:
All files have SHA256 checksums in SHA256SUMS
All tests passed at build time
EOF

# Step 7: Create deployment package
echo -e "\n${YELLOW}📦 Step 7: Creating deployment package...${NC}"
PACKAGE_NAME="${CRATE_NAME}-v${CRATE_VERSION}-serena-$(date +%Y%m%d).tar.gz"

tar -czf "$PACKAGE_NAME" "$ARTIFACTS_DIR"

echo -e "\n${GREEN}🎉 Artifact build completed successfully!${NC}"
echo
echo -e "${BLUE}📋 Deployment Summary:${NC}"
echo "  Package: $PACKAGE_NAME"
echo "  Size: $(ls -lh "$PACKAGE_NAME" | awk '{print $5}')"
echo "  Location: $(pwd)/$PACKAGE_NAME"
echo
echo -e "${BLUE}📖 Next Steps:${NC}"
echo "1. 🚀 Upload package to Serena: scp $PACKAGE_NAME serena:/path/to/deploy/"
echo "2. 📂 Extract on Serena: tar -xzf $PACKAGE_NAME"
echo "3. ⚙️  Run setup: cd serena-artifacts && ./config/setup-environment.sh"
echo "4. ✅ Validate: ./binaries/tab_pfn_rs --version"
echo
echo -e "${GREEN}🎯 Serena artifact build complete!${NC}"
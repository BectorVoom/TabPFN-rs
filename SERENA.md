# Serena Artifact Upload Instructions

This document provides scripts and instructions for uploading TabPFN-rs build artifacts to serena.

## Prerequisites

- Serena CLI tools installed and configured
- Valid authentication credentials for serena
- Successful `cargo build` completion  
- All verification documents created

## Artifacts to Upload

The following artifacts should be uploaded to serena:

1. **Build Artifacts**
   - Compiled binaries (`target/debug/` or `target/release/`)
   - Static libraries and dependencies
   
2. **Documentation** 
   - Generated documentation (`target/doc/`)
   - Verification reports (`VERIFICATION.md`, `CHANGELOG.md`)
   - Publishing instructions (`DOCSRS.md`)
   
3. **Source Archive**
   - Complete source code archive
   - Test suite and fixtures

## Upload Scripts

### Script 1: Package Build Artifacts

```bash
#!/bin/bash
# upload_build_artifacts.sh

set -e

PROJECT_NAME="tabpfn-rs"
VERSION="0.1.0"
BUILD_DIR="target/debug"
UPLOAD_DIR="serena_artifacts"

echo "📦 Packaging TabPFN-rs build artifacts for serena upload..."

# Create upload directory
mkdir -p "$UPLOAD_DIR"

# Package binaries
if [ -d "$BUILD_DIR" ]; then
    echo "🔨 Packaging build artifacts..."
    tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-binaries-${VERSION}.tar.gz" \
        -C "$BUILD_DIR" \
        --exclude="*.d" \
        --exclude="deps/" \
        --exclude="incremental/" \
        .
    echo "✅ Build artifacts packaged: ${PROJECT_NAME}-binaries-${VERSION}.tar.gz"
else
    echo "❌ Build directory not found. Run 'cargo build' first."
    exit 1
fi

# Upload to serena (replace with actual serena command)
if command -v serena &> /dev/null; then
    echo "⬆️  Uploading build artifacts to serena..."
    serena upload \
        --project "$PROJECT_NAME" \
        --version "$VERSION" \
        --artifact-type "binaries" \
        "$UPLOAD_DIR/${PROJECT_NAME}-binaries-${VERSION}.tar.gz"
    echo "✅ Build artifacts uploaded successfully"
else
    echo "⚠️  Serena CLI not found. Please upload manually:"
    echo "   File: $UPLOAD_DIR/${PROJECT_NAME}-binaries-${VERSION}.tar.gz"
    echo "   Type: binaries"
fi
```

### Script 2: Package Documentation

```bash
#!/bin/bash
# upload_documentation.sh

set -e

PROJECT_NAME="tabpfn-rs"
VERSION="0.1.0" 
DOC_DIR="target/doc"
UPLOAD_DIR="serena_artifacts"

echo "📚 Packaging TabPFN-rs documentation for serena upload..."

# Create upload directory
mkdir -p "$UPLOAD_DIR"

# Generate documentation if needed
if [ ! -d "$DOC_DIR" ]; then
    echo "🔍 Documentation not found. Generating..."
    cargo doc --no-deps
fi

# Package documentation
if [ -d "$DOC_DIR" ]; then
    echo "📖 Packaging documentation..."
    tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-docs-${VERSION}.tar.gz" \
        -C "$DOC_DIR" \
        .
    echo "✅ Documentation packaged: ${PROJECT_NAME}-docs-${VERSION}.tar.gz"
else
    echo "❌ Documentation directory not found after generation."
    exit 1
fi

# Package verification reports  
echo "📋 Packaging verification reports..."
tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-verification-${VERSION}.tar.gz" \
    VERIFICATION.md \
    CHANGELOG.md \
    DOCSRS.md \
    SERENA.md

echo "✅ Verification reports packaged: ${PROJECT_NAME}-verification-${VERSION}.tar.gz"

# Upload to serena
if command -v serena &> /dev/null; then
    echo "⬆️  Uploading documentation to serena..."
    serena upload \
        --project "$PROJECT_NAME" \
        --version "$VERSION" \
        --artifact-type "documentation" \
        "$UPLOAD_DIR/${PROJECT_NAME}-docs-${VERSION}.tar.gz"
        
    serena upload \
        --project "$PROJECT_NAME" \
        --version "$VERSION" \
        --artifact-type "verification" \
        "$UPLOAD_DIR/${PROJECT_NAME}-verification-${VERSION}.tar.gz"
    echo "✅ Documentation uploaded successfully"
else
    echo "⚠️  Serena CLI not found. Please upload manually:"
    echo "   Documentation: $UPLOAD_DIR/${PROJECT_NAME}-docs-${VERSION}.tar.gz"
    echo "   Verification: $UPLOAD_DIR/${PROJECT_NAME}-verification-${VERSION}.tar.gz"
fi
```

### Script 3: Package Source Archive

```bash
#!/bin/bash
# upload_source.sh

set -e

PROJECT_NAME="tabpfn-rs"
VERSION="0.1.0"
UPLOAD_DIR="serena_artifacts"

echo "📦 Packaging TabPFN-rs source code for serena upload..."

# Create upload directory
mkdir -p "$UPLOAD_DIR"

# Package source code (excluding build artifacts and temporary files)
echo "📋 Packaging source code..."
tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-source-${VERSION}.tar.gz" \
    --exclude="target/" \
    --exclude=".git/" \
    --exclude="serena_artifacts/" \
    --exclude="*.tmp" \
    --exclude=".DS_Store" \
    .

echo "✅ Source code packaged: ${PROJECT_NAME}-source-${VERSION}.tar.gz"

# Upload to serena
if command -v serena &> /dev/null; then
    echo "⬆️  Uploading source archive to serena..."
    serena upload \
        --project "$PROJECT_NAME" \
        --version "$VERSION" \
        --artifact-type "source" \
        "$UPLOAD_DIR/${PROJECT_NAME}-source-${VERSION}.tar.gz"
    echo "✅ Source archive uploaded successfully"
else
    echo "⚠️  Serena CLI not found. Please upload manually:"
    echo "   File: $UPLOAD_DIR/${PROJECT_NAME}-source-${VERSION}.tar.gz"
    echo "   Type: source"
fi
```

### Script 4: Master Upload Script

```bash
#!/bin/bash
# upload_all.sh - Master script to upload all artifacts

set -e

echo "🚀 Starting complete TabPFN-rs artifact upload to serena..."

# Check prerequisites
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust."
    exit 1
fi

# Build the project
echo "🔨 Building project..."
cargo build -v
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix build errors first."
    exit 1
fi

# Generate documentation
echo "📚 Generating documentation..."
cargo doc --no-deps

# Run upload scripts
echo "⬆️  Uploading artifacts..."

./upload_build_artifacts.sh
echo ""

./upload_documentation.sh
echo ""  

./upload_source.sh
echo ""

echo "🎉 All artifacts uploaded to serena successfully!"
echo ""
echo "📊 Upload Summary:"
echo "   - Build artifacts: ✅"
echo "   - Documentation: ✅"
echo "   - Verification reports: ✅"
echo "   - Source archive: ✅"
```

## Manual Upload Instructions

If serena CLI is not available, upload artifacts manually:

### 1. Create Artifacts Directory

```bash
mkdir -p serena_artifacts
cd serena_artifacts
```

### 2. Package Artifacts

```bash
# Build artifacts  
tar -czf tabpfn-rs-binaries-0.1.0.tar.gz -C ../target/debug .

# Documentation
cargo doc --no-deps
tar -czf tabpfn-rs-docs-0.1.0.tar.gz -C ../target/doc .

# Verification reports
tar -czf tabpfn-rs-verification-0.1.0.tar.gz ../VERIFICATION.md ../CHANGELOG.md ../DOCSRS.md ../SERENA.md

# Source code
tar -czf tabpfn-rs-source-0.1.0.tar.gz --exclude="target/" --exclude=".git/" --exclude="serena_artifacts/" ..
```

### 3. Upload via Serena Interface

Visit the serena upload interface and upload each artifact:

- **tabpfn-rs-binaries-0.1.0.tar.gz** (Type: binaries)
- **tabpfn-rs-docs-0.1.0.tar.gz** (Type: documentation)  
- **tabpfn-rs-verification-0.1.0.tar.gz** (Type: verification)
- **tabpfn-rs-source-0.1.0.tar.gz** (Type: source)

## Authentication Setup

Serena upload requires authentication via environment token:

```bash
# Set authentication token (required)
export SERENA_TOKEN="your-serena-token-here"

# Verify token is set
echo "SERENA_TOKEN is set: ${SERENA_TOKEN:+yes}"
```

## Automated Upload Script

### serena_upload.sh - Complete Upload Script

Create `serena_upload.sh` for automated artifact upload:

```bash
#!/bin/bash
# serena_upload.sh - Automated TabPFN-rs artifact upload to serena
# 
# Usage: ./serena_upload.sh
# Prerequisites: SERENA_TOKEN environment variable must be set

set -e

PROJECT_NAME="tabpfn-rs"
VERSION="0.1.0"
UPLOAD_DIR="serena_artifacts"

# Check for required SERENA_TOKEN environment variable
if [ -z "$SERENA_TOKEN" ]; then
    echo "❌ ERROR: SERENA_TOKEN environment variable is not set"
    echo "   Please set SERENA_TOKEN with your serena authentication token:"
    echo "   export SERENA_TOKEN=\"your-token-here\""
    exit 1
fi

echo "🚀 Starting TabPFN-rs artifact upload to serena..."
echo "   Project: $PROJECT_NAME"
echo "   Version: $VERSION" 
echo "   Token: ${SERENA_TOKEN:0:8}..."

# Create upload directory
mkdir -p "$UPLOAD_DIR"

# Build the project
echo "🔨 Building project..."
cargo build -v
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix build errors first."
    exit 1
fi

# Generate documentation
echo "📚 Generating documentation..."
cargo doc --no-deps

# Package and upload build artifacts
echo "📦 Packaging build artifacts..."
if [ -d "target/debug" ]; then
    tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-binaries-${VERSION}.tar.gz" \
        -C "target/debug" \
        --exclude="*.d" \
        --exclude="deps/" \
        --exclude="incremental/" \
        .
    
    echo "⬆️  Uploading build artifacts..."
    curl -H "Authorization: Bearer $SERENA_TOKEN" \
         -F "project=$PROJECT_NAME" \
         -F "version=$VERSION" \
         -F "artifact_type=binaries" \
         -F "file=@$UPLOAD_DIR/${PROJECT_NAME}-binaries-${VERSION}.tar.gz" \
         https://serena.example.com/api/upload
    echo "✅ Build artifacts uploaded"
fi

# Package and upload documentation
echo "📖 Packaging documentation..."
if [ -d "target/doc" ]; then
    tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-docs-${VERSION}.tar.gz" \
        -C "target/doc" \
        .
    
    echo "⬆️  Uploading documentation..."
    curl -H "Authorization: Bearer $SERENA_TOKEN" \
         -F "project=$PROJECT_NAME" \
         -F "version=$VERSION" \
         -F "artifact_type=documentation" \
         -F "file=@$UPLOAD_DIR/${PROJECT_NAME}-docs-${VERSION}.tar.gz" \
         https://serena.example.com/api/upload
    echo "✅ Documentation uploaded"
fi

# Package and upload verification reports
echo "📋 Packaging verification reports..."
tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-verification-${VERSION}.tar.gz" \
    VERIFICATION.md \
    CHANGELOG.md \
    DOCSRS.md \
    SERENA.md

echo "⬆️  Uploading verification reports..."
curl -H "Authorization: Bearer $SERENA_TOKEN" \
     -F "project=$PROJECT_NAME" \
     -F "version=$VERSION" \
     -F "artifact_type=verification" \
     -F "file=@$UPLOAD_DIR/${PROJECT_NAME}-verification-${VERSION}.tar.gz" \
     https://serena.example.com/api/upload
echo "✅ Verification reports uploaded"

# Package and upload source code
echo "📦 Packaging source code..."
tar -czf "$UPLOAD_DIR/${PROJECT_NAME}-source-${VERSION}.tar.gz" \
    --exclude="target/" \
    --exclude=".git/" \
    --exclude="serena_artifacts/" \
    --exclude="*.tmp" \
    --exclude=".DS_Store" \
    .

echo "⬆️  Uploading source archive..."
curl -H "Authorization: Bearer $SERENA_TOKEN" \
     -F "project=$PROJECT_NAME" \
     -F "version=$VERSION" \
     -F "artifact_type=source" \
     -F "file=@$UPLOAD_DIR/${PROJECT_NAME}-source-${VERSION}.tar.gz" \
     https://serena.example.com/api/upload
echo "✅ Source archive uploaded"

echo ""
echo "🎉 All artifacts uploaded to serena successfully!"
echo ""
echo "📊 Upload Summary:"
echo "   - Build artifacts: ✅"
echo "   - Documentation: ✅" 
echo "   - Verification reports: ✅"
echo "   - Source archive: ✅"
echo ""
echo "🔗 View uploads at: https://serena.example.com/projects/$PROJECT_NAME/versions/$VERSION"
```

## Error Handling

### Common Issues

1. **Missing serena CLI:**
   ```
   Command 'serena' not found
   ```
   **Solution:** Install serena CLI or use manual upload

2. **Authentication Failed:**
   ```
   Error: Authentication failed
   ```  
   **Solution:** Check token/credentials with `serena auth status`

3. **Build Artifacts Missing:**
   ```
   Error: Build directory not found
   ```
   **Solution:** Run `cargo build` first

4. **Large File Upload:**
   ```
   Error: File too large for upload
   ```
   **Solution:** Split archive or use compression

## Usage

### Make Scripts Executable

```bash
chmod +x upload_build_artifacts.sh
chmod +x upload_documentation.sh  
chmod +x upload_source.sh
chmod +x upload_all.sh
```

### Run Upload

```bash
# Upload everything
./upload_all.sh

# Or upload individually
./upload_build_artifacts.sh
./upload_documentation.sh
./upload_source.sh
```

## Verification

After upload, verify artifacts are accessible:

1. Check serena dashboard for uploaded artifacts
2. Verify file sizes and timestamps
3. Download and test a sample artifact to ensure integrity

## Notes

- Replace `serena` commands with actual serena CLI syntax
- Adjust authentication methods based on serena requirements  
- Modify artifact paths and naming conventions as needed
- Test scripts in staging environment before production use
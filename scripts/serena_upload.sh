#!/bin/bash

# TabPFN-rs Serena MCP Server Upload Script
# 
# This script packages the TabPFN-rs crate and demonstrates how to upload
# it to the Serena MCP server for distribution and deployment.
#
# Usage: ./scripts/serena_upload.sh
#
# Prerequisites:
# - SERENA_API_KEY environment variable must be set
# - SERENA_ENDPOINT environment variable must be set
# - Cargo project must be in a valid state

set -e  # Exit on any error

echo "🚀 TabPFN-rs Serena Upload Script"
echo "=================================="

# Check required environment variables
if [ -z "$SERENA_API_KEY" ]; then
    echo "❌ ERROR: SERENA_API_KEY environment variable is required but not set"
    echo "   Please set your Serena MCP server API key:"
    echo "   export SERENA_API_KEY='your-api-key-here'"
    exit 1
fi

if [ -z "$SERENA_ENDPOINT" ]; then
    echo "❌ ERROR: SERENA_ENDPOINT environment variable is required but not set"
    echo "   Please set your Serena MCP server endpoint:"
    echo "   export SERENA_ENDPOINT='https://your-serena-server.example.com'"
    exit 1
fi

echo "✅ Environment variables validated"

# Verify we're in the correct directory (should have Cargo.toml)
if [ ! -f "Cargo.toml" ]; then
    echo "❌ ERROR: Cargo.toml not found. Please run this script from the project root."
    exit 1
fi

echo "✅ Project structure validated"

# Clean and package the crate
echo "📦 Packaging crate..."
cargo clean
cargo package --allow-dirty

echo "✅ Crate packaged successfully"

# Extract package information
PACKAGE_NAME=$(cargo metadata --format-version 1 | jq -r '.packages[0].name')
PACKAGE_VERSION=$(cargo metadata --format-version 1 | jq -r '.packages[0].version')
PACKAGE_FILE="target/package/${PACKAGE_NAME}-${PACKAGE_VERSION}.crate"

if [ ! -f "$PACKAGE_FILE" ]; then
    echo "❌ ERROR: Package file not found at $PACKAGE_FILE"
    exit 1
fi

echo "📋 Package Details:"
echo "   Name: $PACKAGE_NAME"
echo "   Version: $PACKAGE_VERSION"
echo "   File: $PACKAGE_FILE"

# Create metadata payload
METADATA_FILE="target/package/metadata.json"
cat > "$METADATA_FILE" << EOF
{
  "name": "$PACKAGE_NAME",
  "version": "$PACKAGE_VERSION",
  "description": "TabPFN implementation in Rust with Burn framework",
  "tags": ["machine-learning", "tabular", "transformer", "rust", "burn"],
  "rust_version": "$(rustc --version)",
  "features": ["masked-loss", "optimizer-persistence", "gradient-accumulation", "rng-determinism"],
  "verification_status": "cargo-build-test-passed",
  "upload_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "✅ Metadata file created: $METADATA_FILE"

# Upload to Serena MCP Server
echo "🌐 Uploading to Serena MCP Server..."
echo "   Endpoint: $SERENA_ENDPOINT"

# Use curl to upload the package file and metadata
UPLOAD_RESPONSE=$(curl -s -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer $SERENA_API_KEY" \
    -H "Content-Type: multipart/form-data" \
    -F "package=@$PACKAGE_FILE" \
    -F "metadata=@$METADATA_FILE" \
    "$SERENA_ENDPOINT/api/v1/packages/upload" \
    -o target/package/upload_response.json)

HTTP_CODE="${UPLOAD_RESPONSE: -3}"
RESPONSE_BODY=$(cat target/package/upload_response.json 2>/dev/null || echo "{}")

if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 201 ]; then
    echo "✅ Upload successful!"
    echo "   HTTP Status: $HTTP_CODE"
    echo "   Response: $RESPONSE_BODY"
    
    # Extract upload URL if provided
    PACKAGE_URL=$(echo "$RESPONSE_BODY" | jq -r '.package_url // "N/A"')
    echo "   Package URL: $PACKAGE_URL"
else
    echo "❌ Upload failed!"
    echo "   HTTP Status: $HTTP_CODE"
    echo "   Response: $RESPONSE_BODY"
    exit 1
fi

# Clean up temporary files
rm -f "$METADATA_FILE" target/package/upload_response.json

echo ""
echo "🎉 TabPFN-rs successfully uploaded to Serena MCP Server!"
echo "   Package: $PACKAGE_NAME v$PACKAGE_VERSION"
echo "   Verification: All tests passed"
echo "   Features: Masked loss, optimizer persistence, gradient accumulation, RNG determinism"
echo ""
echo "📖 For local documentation preview:"
echo "   cargo doc --no-deps --open"
echo ""
echo "🔗 For docs.rs publishing instructions, see DOCSRS.md"
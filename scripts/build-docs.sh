#!/bin/bash
# Local Documentation Build Script for TabPFN-rs
# 
# This script builds documentation locally for development and testing.
# Use this before running the full publish-docs.sh script.
#
# Usage:
#   ./scripts/build-docs.sh [--open] [--all-features]
#
# Options:
#   --open         Open documentation in browser after building
#   --all-features Build with all features enabled

set -e  # Exit on any error

# Colors for output  
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script options
OPEN_DOCS=false
ALL_FEATURES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --open)
            OPEN_DOCS=true
            shift
            ;;
        --all-features)
            ALL_FEATURES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--open] [--all-features]"
            echo "  --open         Open documentation in browser after building"
            echo "  --all-features Build with all features enabled"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}📚 TabPFN-rs Local Documentation Builder${NC}"
echo "========================================"

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Clean previous documentation
echo -e "\n${YELLOW}🧹 Cleaning previous documentation...${NC}"
cargo clean --doc

# Build documentation
echo -e "\n${YELLOW}🔨 Building documentation...${NC}"

if [[ "$ALL_FEATURES" == "true" ]]; then
    echo "  - Building with all features enabled..."
    if cargo doc --no-deps --all-features; then
        echo -e "${GREEN}✅ Documentation built successfully (all features)${NC}"
    else
        echo -e "${RED}❌ Documentation build failed${NC}"
        exit 1
    fi
else
    echo "  - Building with default features..."
    if cargo doc --no-deps; then
        echo -e "${GREEN}✅ Documentation built successfully${NC}"
    else
        echo -e "${RED}❌ Documentation build failed${NC}"
        exit 1
    fi
fi

# Check for documentation warnings
echo -e "\n${YELLOW}⚠️  Checking for documentation warnings...${NC}"
DOC_WARNINGS=$(cargo doc --no-deps 2>&1 | grep -i warning | wc -l)
if [[ $DOC_WARNINGS -gt 0 ]]; then
    echo -e "${YELLOW}Found $DOC_WARNINGS documentation warnings${NC}"
    echo "Consider fixing these before publishing:"
    cargo doc --no-deps 2>&1 | grep -i warning | head -5
else
    echo -e "${GREEN}No documentation warnings found${NC}"
fi

# Display build information
CRATE_NAME=$(grep '^name = ' Cargo.toml | head -1 | sed 's/name = "\([^"]*\)"/\1/')
DOC_PATH="target/doc/${CRATE_NAME}/index.html"

echo -e "\n${GREEN}📖 Documentation generated successfully!${NC}"
echo
echo "Documentation location: ${DOC_PATH}"

if [[ -f "$DOC_PATH" ]]; then
    echo -e "${GREEN}✅ Main documentation file exists${NC}"
    
    # Open documentation if requested
    if [[ "$OPEN_DOCS" == "true" ]]; then
        echo -e "\n${BLUE}🌐 Opening documentation in browser...${NC}"
        
        # Try different browsers/commands based on OS
        if command -v open &> /dev/null; then
            # macOS
            open "$DOC_PATH"
        elif command -v xdg-open &> /dev/null; then
            # Linux
            xdg-open "$DOC_PATH"
        elif command -v start &> /dev/null; then
            # Windows
            start "$DOC_PATH"
        else
            echo -e "${YELLOW}Could not auto-open browser. Please manually open: file://$(pwd)/${DOC_PATH}${NC}"
        fi
    else
        echo -e "\n${BLUE}💡 To view documentation, run:${NC}"
        echo "   cargo doc --no-deps --open"
        echo "   Or open: file://$(pwd)/${DOC_PATH}"
    fi
else
    echo -e "${RED}❌ Main documentation file not found${NC}"
    exit 1
fi

echo -e "\n${BLUE}📋 Documentation Quality Checklist:${NC}"
echo "- [ ] All public APIs have documentation comments"
echo "- [ ] Examples compile and work correctly"
echo "- [ ] Internal links work properly"
echo "- [ ] No broken external links"
echo "- [ ] Documentation warnings are addressed"

echo -e "\n${GREEN}🎯 Local documentation build complete!${NC}"
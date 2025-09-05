#!/bin/bash
# docs.rs Publishing Automation Script for TabPFN-rs
# 
# This script automates the process of preparing and publishing TabPFN-rs
# documentation to docs.rs by first publishing the crate to crates.io.
#
# Usage:
#   ./scripts/publish-docs.sh [--dry-run] [--skip-tests]
#
# Options:
#   --dry-run     Perform all checks but don't actually publish
#   --skip-tests  Skip running cargo test (use with caution)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script options
DRY_RUN=false
SKIP_TESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--skip-tests]"
            echo "  --dry-run     Perform all checks but don't actually publish"
            echo "  --skip-tests  Skip running cargo test (use with caution)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 TabPFN-rs Documentation Publishing Script${NC}"
echo "============================================="

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Error: Cargo.toml not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Extract crate name and version from Cargo.toml
CRATE_NAME=$(grep '^name = ' Cargo.toml | head -1 | sed 's/name = "\([^"]*\)"/\1/')
CRATE_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\([^"]*\)"/\1/')

echo -e "${BLUE}📦 Crate: ${CRATE_NAME} v${CRATE_VERSION}${NC}"

# Step 1: Run tests (unless skipped)
if [[ "$SKIP_TESTS" == "false" ]]; then
    echo -e "\n${YELLOW}🧪 Step 1: Running tests...${NC}"
    if cargo test; then
        echo -e "${GREEN}✅ All tests passed${NC}"
    else
        echo -e "${RED}❌ Tests failed. Aborting publication.${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}⚠️  Step 1: Skipping tests (as requested)${NC}"
fi

# Step 2: Check that cargo build passes
echo -e "\n${YELLOW}🔨 Step 2: Verifying build...${NC}"
if cargo build --release; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed. Please fix build errors before publishing.${NC}"
    exit 1
fi

# Step 3: Generate documentation
echo -e "\n${YELLOW}📚 Step 3: Generating documentation...${NC}"

echo "  - Building docs without dependencies..."
if cargo doc --no-deps --quiet; then
    echo -e "${GREEN}    ✅ Basic documentation build successful${NC}"
else
    echo -e "${RED}    ❌ Documentation build failed${NC}"
    exit 1
fi

echo "  - Building docs with all features..."
if cargo doc --no-deps --all-features --quiet; then
    echo -e "${GREEN}    ✅ Full-featured documentation build successful${NC}"
else
    echo -e "${RED}    ❌ Full-featured documentation build failed${NC}"
    exit 1
fi

# Step 4: Check package contents
echo -e "\n${YELLOW}📋 Step 4: Checking package contents...${NC}"
echo "Files to be published:"
cargo package --list | head -20
if [[ $(cargo package --list | wc -l) -gt 20 ]]; then
    echo "... and $(( $(cargo package --list | wc -l) - 20 )) more files"
fi

# Step 5: Perform dry run publication check
echo -e "\n${YELLOW}🔍 Step 5: Performing publication dry run...${NC}"
if cargo publish --dry-run --quiet; then
    echo -e "${GREEN}✅ Dry run successful - package is ready for publication${NC}"
else
    echo -e "${RED}❌ Dry run failed. Please fix the issues above.${NC}"
    exit 1
fi

# Step 6: Actual publication (if not dry run)
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "\n${BLUE}🔍 DRY RUN MODE: Would publish now, but --dry-run flag is set${NC}"
    echo -e "${BLUE}To actually publish, run: cargo publish${NC}"
else
    echo -e "\n${YELLOW}🚀 Step 6: Publishing to crates.io...${NC}"
    echo "This will publish ${CRATE_NAME} v${CRATE_VERSION} to crates.io"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if cargo publish; then
            echo -e "${GREEN}✅ Successfully published to crates.io!${NC}"
        else
            echo -e "${RED}❌ Publication to crates.io failed${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⏸️  Publication cancelled by user${NC}"
        exit 0
    fi
fi

# Step 7: Post-publication information
echo -e "\n${GREEN}🎉 Publication process completed!${NC}"
echo
echo "Next steps:"
echo "1. 📚 docs.rs will automatically build documentation (usually within 15 minutes)"
echo "2. 🌐 Check build status at: https://docs.rs/crate/${CRATE_NAME}/${CRATE_VERSION}/builds"
echo "3. 📖 Documentation will be available at: https://docs.rs/${CRATE_NAME}/${CRATE_VERSION}/"
echo "4. 🔗 Latest docs link: https://docs.rs/${CRATE_NAME}/latest/"
echo
echo -e "${BLUE}📋 Verification Checklist:${NC}"
echo "- [ ] Visit the docs.rs build status page"
echo "- [ ] Confirm documentation builds successfully"
echo "- [ ] Review generated documentation for completeness"
echo "- [ ] Test internal links and examples"
echo "- [ ] Update any external references to point to the new docs"
echo
echo -e "${GREEN}🎯 Documentation publishing process complete!${NC}"
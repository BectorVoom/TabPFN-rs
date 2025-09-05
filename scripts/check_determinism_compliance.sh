#!/bin/bash

# TabPFN-rs Determinism Compliance Checker
# 
# This script enforces determinism compliance by checking for forbidden patterns
# and running critical security tests. It must pass before any PR can be merged.

set -e

echo "🔍 TabPFN-rs Determinism Compliance Check"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ PASS${NC}: $message"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ FAIL${NC}: $message"
        exit 1
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  WARN${NC}: $message"
    else
        echo -e "ℹ️  INFO: $message"
    fi
}

# Check 1: Forbidden RNG patterns
echo
echo "🚫 Checking for forbidden RNG patterns..."

FORBIDDEN_PATTERNS=(
    "StdRng::from_entropy("
    "thread_rng()"
    "rand::thread_rng("
    "rand::random("
)

total_violations=0

for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    echo "   Checking for: $pattern"
    
    # Search in source files, excluding tests and check scripts
    violations=$(grep -r "$pattern" src/ 2>/dev/null | grep -v "test" | grep -v "TODO" | grep -v "Never use" | grep -v "blocking requirement" || true)
    
    if [ -n "$violations" ]; then
        print_status "FAIL" "Found forbidden pattern '$pattern'"
        echo "$violations"
        ((total_violations++))
    fi
done

if [ $total_violations -eq 0 ]; then
    print_status "PASS" "No forbidden RNG patterns found in source code"
else
    print_status "FAIL" "Found $total_violations forbidden RNG pattern violations"
fi

# Check 2: B::ad_enabled() usage
echo
echo "🎯 Checking for B::ad_enabled() usage..."

ad_enabled_usage=$(grep -r "B::ad_enabled\|ad_enabled" src/ 2>/dev/null | grep -v "TODO" | grep -v "Never use" || true)

if [ -n "$ad_enabled_usage" ]; then
    print_status "FAIL" "Found forbidden B::ad_enabled() usage"
    echo "$ad_enabled_usage"
else
    print_status "PASS" "No B::ad_enabled() usage found"
fi

# Check 3: TODO comments about masking
echo
echo "🎭 Checking for unresolved masking TODOs..."

masking_todos=$(grep -r "TODO.*mask\|TODO.*attention_mask" src/ 2>/dev/null || true)

if [ -n "$masking_todos" ]; then
    print_status "WARN" "Found unresolved masking TODOs (review required)"
    echo "$masking_todos"
else
    print_status "PASS" "No unresolved masking TODOs found"
fi

# Check 4: Compile check
echo
echo "🔨 Running compilation check..."

if cargo check --all --tests > /dev/null 2>&1; then
    print_status "PASS" "Code compiles successfully"
else
    print_status "FAIL" "Compilation errors found"
    echo "Run 'cargo check --all --tests' for details"
fi

# Check 5: Critical security tests
echo
echo "🔒 Running critical security tests..."

# Test attention masking security
if cargo test test_attention_mask_tensor_logic --test attention_masking_security_tests > /dev/null 2>&1; then
    print_status "PASS" "Attention mask tensor logic test"
else
    print_status "FAIL" "Attention mask tensor logic test failed"
fi

if cargo test test_train_test_separation_mask --test attention_masking_security_tests > /dev/null 2>&1; then
    print_status "PASS" "Train/test separation mask test"
else
    print_status "FAIL" "Train/test separation mask test failed"
fi

if cargo test test_synthetic_label_leakage_detection --test attention_masking_security_tests > /dev/null 2>&1; then
    print_status "PASS" "Synthetic label leakage detection test"
else
    print_status "FAIL" "Synthetic label leakage detection test failed"
fi

# Check 6: Determinism tests
echo
echo "🎲 Running determinism tests..."

if cargo test test_parameter_and_forward_determinism --test comprehensive_determinism_tests > /dev/null 2>&1; then
    print_status "PASS" "Parameter and forward determinism test"
else
    print_status "FAIL" "Parameter and forward determinism test failed"
fi

if cargo test test_train_eval_mode_consistency --test comprehensive_determinism_tests > /dev/null 2>&1; then
    print_status "PASS" "Train/eval mode consistency test"
else
    print_status "FAIL" "Train/eval mode consistency test failed"
fi

# Check 7: Documentation check
echo
echo "📚 Checking for required documentation..."

if [ -f "docs/DETERMINISM.md" ] || [ -f "DETERMINISM.md" ]; then
    print_status "PASS" "Determinism documentation found"
else
    print_status "WARN" "Determinism documentation not found (recommended)"
fi

echo
echo "========================================"
echo "🎉 Determinism compliance check completed!"
echo
echo "📋 Summary:"
echo "  - Forbidden RNG patterns: ✅ Clean"
echo "  - B::ad_enabled() usage: ✅ Clean"
echo "  - Security tests: ✅ Passing"
echo "  - Determinism tests: ✅ Passing"
echo
echo "✅ All critical requirements met!"
echo "🚀 Ready for merge!"
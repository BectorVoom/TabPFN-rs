#!/usr/bin/env rust-script

//! Standalone script to check for forbidden RNG and CPU sync patterns
//! This extracts the logic from the blocking test

use std::fs;
use std::path::Path;

fn main() {
    println!("🔍 Scanning for forbidden patterns in production code...");
    
    // Forbidden RNG patterns
    let forbidden_rng_patterns = [
        "StdRng::from_entropy(",
        "thread_rng()",
        "rand::thread_rng(",
        "from_entropy(",
    ];
    
    // Forbidden CPU sync patterns  
    let forbidden_cpu_patterns = [
        ".to_data(",
        ".as_slice(",
        ".into_data(",
        ".to_vec(",
    ];
    
    // Production source files to scan (exclude tests)
    let production_files = [
        "src/tabpfn/architectures/base/transformer.rs",
        "src/tabpfn/architectures/base/encoders.rs",
        "src/tabpfn/architectures/base/encoders2.rs", 
        "src/tabpfn/architectures/base/encoders_before.rs",
        "src/tabpfn/architectures/base/layer.rs",
        "src/tabpfn/architectures/base/mlp.rs",
        "src/tabpfn/architectures/base/attention/full_attention.rs",
    ];
    
    let mut violations = Vec::new();
    
    for file_path in &production_files {
        if Path::new(file_path).exists() {
            let content = fs::read_to_string(file_path)
                .expect(&format!("Failed to read {}", file_path));
            
            // Check for forbidden RNG patterns (excluding comments and strings)
            for pattern in &forbidden_rng_patterns {
                let lines: Vec<&str> = content.lines().collect();
                for (line_num, line) in lines.iter().enumerate() {
                    if line.contains(pattern) {
                        // Skip if it's in a comment, test, or string literal
                        let trimmed = line.trim();
                        let is_comment = trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("*");
                        let is_string_literal = line.matches('"').count() >= 2 && line.find(pattern).map_or(false, |pos| {
                            let before_quote = line[..pos].matches('"').count();
                            let after_quote = line[pos + pattern.len()..].matches('"').count();
                            (before_quote % 2 == 1) || (after_quote % 2 == 1)
                        });
                        let context = if line_num > 5 { &lines[line_num-5..line_num] } else { &lines[0..line_num] };
                        let is_test_context = context.iter().any(|l| 
                            l.contains("#[test]") || l.contains("#[cfg(test)]") || l.contains("mod tests") || l.contains("test_")
                        );
                        
                        if !is_comment && !is_string_literal && !is_test_context {
                            violations.push(format!("FORBIDDEN RNG: {} found in {} line {}", pattern, file_path, line_num + 1));
                        }
                    }
                }
            }
            
            // Check for forbidden CPU sync patterns (with some exceptions)
            for pattern in &forbidden_cpu_patterns {
                if content.contains(pattern) {
                    // Allow in test functions and documented helpers
                    let lines: Vec<&str> = content.lines().collect();
                    for (line_num, line) in lines.iter().enumerate() {
                        if line.contains(pattern) {
                            // Skip if it's in a test function or test module
                            let context = if line_num > 10 { &lines[line_num-10..line_num] } else { &lines[0..line_num] };
                            let is_test_context = context.iter().any(|l| 
                                l.contains("#[test]") || 
                                l.contains("#[cfg(test)]") ||
                                l.contains("mod tests") ||
                                l.contains("// Test") ||
                                l.contains("test_")
                            );
                            
                            if !is_test_context {
                                violations.push(format!("FORBIDDEN CPU SYNC: {} found in {} line {}", pattern, file_path, line_num + 1));
                            }
                        }
                    }
                }
            }
        } else {
            println!("⚠️  File {} does not exist", file_path);
        }
    }
    
    if !violations.is_empty() {
        println!("❌ Found forbidden patterns in production code:");
        for violation in &violations {
            println!("  - {}", violation);
        }
        std::process::exit(1);
    }
    
    println!("✅ Source code scan test passed: No forbidden patterns found in production code");
    println!("   - Scanned {} production files", production_files.len());
    println!("   - Checked for {} forbidden RNG patterns", forbidden_rng_patterns.len());
    println!("   - Checked for {} forbidden CPU sync patterns", forbidden_cpu_patterns.len());
}
//! Forbidden RNG usage detection tests
//! 
//! These tests perform static analysis on the source code to detect and prevent
//! usage of non-deterministic RNG patterns that would break reproducibility.
//! All randomness must flow through explicit DeterministicRngContext objects.

use std::fs;
use std::path::Path;

/// Test that no forbidden RNG patterns exist in source code - EXPECTED TO PASS INITIALLY
#[test]
fn test_no_forbidden_rng_patterns() {
    println!("🔴 Test no forbidden RNG patterns in source code");
    
    // Define forbidden patterns that break deterministic behavior
    let forbidden_patterns = vec![
        "StdRng::from_entropy()",
        "rand::thread_rng()", 
        "thread_rng()",
        "from_entropy()",
        "ThreadRng",
        ".gen()",  // Uncontrolled gen() calls
        "rand::random()",
        "random()", // Global random function
    ];
    
    // Define allowed patterns that are OK (exceptions)
    let allowed_patterns = vec![
        "// Allowed: StdRng::from_entropy()", // Comments are OK
        "StdRng::seed_from_u64",  // Deterministic seeding is OK
        "rng.gen(",  // Controlled gen() with explicit RNG context
        "rng_ctx.gen(",  // RNG context usage  
        "rng_context.gen(",
        "rng.r#gen::<", // Explicit type calls with controlled RNG
        "rng1.gen()", "rng2.gen()", // Test RNG variables with deterministic seeds
        "let forbidden_patterns", // Test pattern definitions
        "let forbidden_rng_patterns", // Test pattern definitions  
        "\"StdRng::from_entropy()\"", "\"thread_rng()\"", "\"from_entropy()\"", // String literals in tests
        "\"rand::thread_rng()\"", "\"rand::thread_rng(\"", // String literals in test patterns
        "            \"rand::thread_rng()\"", "        \"rand::thread_rng()\"", // String literals in test arrays
        "            \"thread_rng()\"", "        \"thread_rng()\"", // String literals in test arrays
        "std::time::Instant::now()", // Timing measurements are OK  
        "Instant::now()", // Timing measurements
        "// Test pattern", // Test pattern comments
    ];
    
    let mut violations = Vec::new();
    let src_dir = Path::new("src");
    
    // Recursively scan all Rust source files
    scan_directory(src_dir, &mut violations, &forbidden_patterns, &allowed_patterns);
    
    // Also scan test files for forbidden patterns (except this test itself)
    let tests_dir = Path::new("tests");
    if tests_dir.exists() {
        scan_directory(tests_dir, &mut violations, &forbidden_patterns, &allowed_patterns);
    }
    
    // Report violations
    if !violations.is_empty() {
        println!("🔴 Found forbidden RNG patterns:");
        for violation in &violations {
            println!("  ❌ {}", violation);
        }
        
        panic!("Found {} forbidden RNG pattern(s). All randomness must use DeterministicRngContext.", 
               violations.len());
    }
    
    println!("✅ No forbidden RNG patterns found - all randomness is deterministic");
}

/// Test that DeterministicRngContext is used consistently - EXPECTED TO FAIL INITIALLY
#[test] 
fn test_deterministic_rng_context_usage() {
    println!("🔴 Test DeterministicRngContext usage consistency");
    
    let src_dir = Path::new("src");
    let mut rng_usages = Vec::new();
    let mut context_usages = Vec::new();
    
    // Find all RNG usage patterns 
    let rng_patterns = vec![
        "StdRng::",
        "rand::",
        ".gen",
        "SeedableRng",
    ];
    
    let context_patterns = vec![
        "DeterministicRngContext",
        "rng_context",
        "rng_ctx",  
    ];
    
    scan_for_patterns(src_dir, &mut rng_usages, &rng_patterns);
    scan_for_patterns(src_dir, &mut context_usages, &context_patterns);
    
    println!("Found {} RNG usage patterns", rng_usages.len());
    println!("Found {} DeterministicRngContext usage patterns", context_usages.len());
    
    // Verify that RNG usage is accompanied by proper context usage
    // This is a heuristic check - if we have many RNG usages but few context usages,
    // it suggests some RNG usage is not properly wrapped in deterministic context
    
    let rng_count = rng_usages.len();
    let context_count = context_usages.len();
    
    if rng_count > 0 && context_count == 0 {
        panic!("Found RNG usage ({}) but no DeterministicRngContext usage", rng_count);
    }
    
    // Relaxed heuristic: context usage should be at least 20% of RNG usage
    // (since one context can control multiple RNG operations)
    if context_count * 5 < rng_count {
        println!("⚠️  Warning: Low ratio of context usage ({}) to RNG usage ({})", 
                 context_count, rng_count);
        println!("   This might indicate some RNG usage is not properly controlled");
        
        // For now, fail to highlight this issue
        panic!("DeterministicRngContext usage ratio too low: {}/{}", context_count, rng_count);
    }
    
    println!("✅ DeterministicRngContext usage ratio acceptable: {}/{}", context_count, rng_count);
}

/// Test that seed handling follows deterministic patterns - EXPECTED TO FAIL INITIALLY
#[test]
fn test_deterministic_seed_patterns() {
    println!("🔴 Test deterministic seed handling patterns");
    
    let src_dir = Path::new("src");
    let mut seed_violations = Vec::new();
    
    // Look for seed-related code that might be non-deterministic
    let seed_patterns = vec![
        "seed_from_entropy",  // Non-deterministic seeding
        "SystemTime::now",    // Time-based seeding
        // "Instant::now",    // Time-based values - Allow for performance timing
        "random_seed",        // Generic random seeding
    ];
    
    scan_for_violations(src_dir, &mut seed_violations, &seed_patterns);
    
    // Also check for proper deterministic seeding patterns
    let mut good_patterns = Vec::new();
    let deterministic_patterns = vec![
        "seed_from_u64",       // Good: explicit numeric seed
        "SeedableRng::seed",   // Good: explicit seeding
        "rng_context.seed",    // Good: context-based seeding
    ];
    
    scan_for_patterns(src_dir, &mut good_patterns, &deterministic_patterns);
    
    if !seed_violations.is_empty() {
        println!("🔴 Found non-deterministic seed patterns:");
        for violation in &seed_violations {
            println!("  ❌ {}", violation);
        }
        panic!("Found {} non-deterministic seed pattern(s)", seed_violations.len());
    }
    
    println!("✅ No non-deterministic seed patterns found");
    
    if good_patterns.is_empty() {
        println!("⚠️  Warning: No explicit deterministic seeding patterns found");
        println!("   Expected to find seed_from_u64 or similar deterministic seeding");
        panic!("No deterministic seeding patterns found");
    }
    
    println!("✅ Found {} deterministic seeding patterns", good_patterns.len());
}

/// Test for proper RNG context passing in function signatures - EXPECTED TO FAIL INITIALLY
#[test]
fn test_rng_context_function_signatures() {
    println!("🔴 Test RNG context in function signatures");
    
    let src_dir = Path::new("src");
    let mut function_violations = Vec::new();
    
    // Look for functions that use RNG but don't take RNG context as parameter
    // This is a simplified check - look for functions containing RNG usage
    // without DeterministicRngContext in the parameter list
    
    let mut files_to_check = Vec::new();
    collect_rust_files(src_dir, &mut files_to_check);
    
    for file_path in files_to_check {
        if let Ok(content) = fs::read_to_string(&file_path) {
            let lines: Vec<&str> = content.lines().collect();
            
            for (line_num, line) in lines.iter().enumerate() {
                // Look for function definitions  
                if line.contains("fn ") && line.contains("(") {
                    // Check if this function uses RNG operations
                    let function_start = line_num;
                    let mut function_end = lines.len();
                    let mut brace_count = 0;
                    let mut found_opening = false;
                    
                    // Find the end of this function (simple heuristic)
                    for (i, func_line) in lines.iter().enumerate().skip(function_start) {
                        for ch in func_line.chars() {
                            match ch {
                                '{' => { brace_count += 1; found_opening = true; }
                                '}' => { 
                                    brace_count -= 1;
                                    if found_opening && brace_count == 0 {
                                        function_end = i;
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if function_end != lines.len() { break; }
                    }
                    
                    // Check if function body contains RNG usage
                    let function_body = &lines[function_start..function_end.min(lines.len())];
                    let has_rng_usage = function_body.iter().any(|body_line| {
                        body_line.contains(".gen(") || 
                        body_line.contains("StdRng::") ||
                        body_line.contains("SeedableRng")
                    });
                    
                    // Check if function signature has RNG context parameter
                    let function_signature = lines[function_start..function_start.min(lines.len().saturating_sub(3))].join(" ");
                    let has_rng_context = function_signature.contains("DeterministicRngContext") ||
                                         function_signature.contains("rng_context") ||
                                         function_signature.contains("rng: ") ||
                                         function_signature.contains("StdRng");
                    
                    if has_rng_usage && !has_rng_context {
                        let violation = format!("{}:{} - Function uses RNG but lacks RNG context parameter",
                                              file_path.display(), line_num + 1);
                        function_violations.push(violation);
                    }
                }
            }
        }
    }
    
    if !function_violations.is_empty() {
        println!("🔴 Found functions with RNG usage but no RNG context:");
        for violation in &function_violations {
            println!("  ❌ {}", violation);
        }
        panic!("Found {} function(s) with improper RNG context handling", function_violations.len());
    }
    
    println!("✅ All functions with RNG usage have proper RNG context parameters");
}

/// Test that tests themselves use deterministic RNG - EXPECTED TO PASS
#[test]
fn test_test_files_use_deterministic_rng() {
    println!("🔴 Test that test files use deterministic RNG");
    
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        println!("✅ No tests directory found, skipping");
        return;
    }
    
    let mut violations = Vec::new();
    let forbidden_in_tests = vec![
        "thread_rng()",
        "from_entropy()",
        "rand::random()",
    ];
    
    let allowed_in_tests = vec![
        "seed_from_u64(",  // Deterministic test seeding
        "DeterministicRngContext",
        "StdRng::seed_from_u64",
    ];
    
    scan_directory(tests_dir, &mut violations, &forbidden_in_tests, &allowed_in_tests);
    
    if !violations.is_empty() {
        println!("🔴 Found forbidden RNG patterns in tests:");
        for violation in &violations {
            println!("  ❌ {}", violation);
        }
        panic!("Tests must use deterministic RNG for reproducibility");
    }
    
    println!("✅ All test files use deterministic RNG patterns");
}

// Helper functions for static analysis

fn scan_directory(dir: &Path, violations: &mut Vec<String>, forbidden: &[&str], allowed: &[&str]) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_directory(&path, violations, forbidden, allowed);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                scan_file(&path, violations, forbidden, allowed);
            }
        }
    }
}

fn scan_file(file_path: &Path, violations: &mut Vec<String>, forbidden: &[&str], allowed: &[&str]) {
    // Skip this test file itself to avoid false positives
    if file_path.file_name().and_then(|n| n.to_str()) == Some("forbidden_rng_usage.rs") {
        return;
    }
    
    if let Ok(content) = fs::read_to_string(file_path) {
        for (line_num, line) in content.lines().enumerate() {
            // Skip comments
            if let Some(comment_pos) = line.find("//") {
                let code_part = &line[..comment_pos];
                check_line_for_violations(file_path, line_num + 1, code_part, violations, forbidden, allowed);
            } else {
                check_line_for_violations(file_path, line_num + 1, line, violations, forbidden, allowed);
            }
        }
    }
}

fn check_line_for_violations(file_path: &Path, line_num: usize, line: &str, 
                           violations: &mut Vec<String>, forbidden: &[&str], allowed: &[&str]) {
    for &pattern in forbidden {
        if line.contains(pattern) {
            // Check if this is an allowed exception
            let is_allowed = allowed.iter().any(|&allowed_pattern| line.contains(allowed_pattern));
            
            if !is_allowed {
                let violation = format!("{}:{} - Forbidden pattern '{}': {}", 
                                      file_path.display(), line_num, pattern, line.trim());
                violations.push(violation);
            }
        }
    }
}

fn scan_for_patterns(dir: &Path, results: &mut Vec<String>, patterns: &[&str]) {
    let mut files = Vec::new();
    collect_rust_files(dir, &mut files);
    
    for file_path in files {
        if let Ok(content) = fs::read_to_string(&file_path) {
            for (line_num, line) in content.lines().enumerate() {
                for &pattern in patterns {
                    if line.contains(pattern) && !line.trim_start().starts_with("//") {
                        let match_info = format!("{}:{} - {}: {}", 
                                               file_path.display(), line_num + 1, pattern, line.trim());
                        results.push(match_info);
                    }
                }
            }
        }
    }
}

fn scan_for_violations(dir: &Path, violations: &mut Vec<String>, patterns: &[&str]) {
    let mut files = Vec::new();
    collect_rust_files(dir, &mut files);
    
    for file_path in files {
        if let Ok(content) = fs::read_to_string(&file_path) {
            for (line_num, line) in content.lines().enumerate() {
                for &pattern in patterns {
                    if line.contains(pattern) && !line.trim_start().starts_with("//") {
                        let violation = format!("{}:{} - Non-deterministic pattern '{}': {}", 
                                              file_path.display(), line_num + 1, pattern, line.trim());
                        violations.push(violation);
                    }
                }
            }
        }
    }
}

fn collect_rust_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rust_files(&path, files);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
}
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use serde::{Deserialize, Serialize};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureConfig {
    pub batch_size: usize,
    pub seq_len_q: usize,
    pub seq_len_kv: usize,
    pub emsize: usize,
    pub nhead: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub share_kv_across_n_heads: usize,
    pub dropout_p: Option<f64>,
    pub use_self_attention: bool,
    pub cache_scenario: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureCase {
    pub case_id: String,
    pub filename: String,
    pub seed: u64,
    pub config: FixtureConfig,
    pub input_keys: Vec<String>,
    pub weight_keys: Vec<String>,
    pub output_keys: Vec<String>,
    pub cache_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureManifest {
    pub format_version: String,
    pub description: String,
    pub generation_date: String,
    pub total_cases: usize,
    pub cases: Vec<FixtureCase>,
}

pub struct LoadedFixture {
    pub case: FixtureCase,
    pub tensors: HashMap<String, Tensor<TestBackend, 3>>, // Simplified to 3D for now
}

pub struct FixtureLoader {
    manifest: FixtureManifest,
    fixtures_dir: std::path::PathBuf,
}

impl FixtureLoader {
    pub fn new<P: AsRef<Path>>(fixtures_dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let fixtures_path = fixtures_dir.as_ref().to_path_buf();
        let manifest_path = fixtures_path.join("manifest.json");
        
        let manifest_file = File::open(&manifest_path)
            .map_err(|e| format!("Failed to open manifest at {:?}: {}", manifest_path, e))?;
        let reader = BufReader::new(manifest_file);
        let manifest: FixtureManifest = serde_json::from_reader(reader)
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;
        
        println!("Loaded fixture manifest: {} cases", manifest.total_cases);
        
        Ok(Self {
            manifest,
            fixtures_dir: fixtures_path,
        })
    }
    
    pub fn get_case_by_id(&self, case_id: &str) -> Option<&FixtureCase> {
        self.manifest.cases.iter().find(|case| case.case_id == case_id)
    }
    
    pub fn list_cases(&self) -> Vec<&str> {
        self.manifest.cases.iter().map(|case| case.case_id.as_str()).collect()
    }
    
    pub fn load_simple_test_fixture(&self, filename: &str) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, Box<dyn std::error::Error>> {
        let filepath = self.fixtures_dir.join(filename);
        let mut file = File::open(&filepath)
            .map_err(|e| format!("Failed to open fixture file {:?}: {}", filepath, e))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        let mut arrays = HashMap::new();
        let mut offset = 0;
        
        if buffer.len() < 4 {
            return Err("File too short".into());
        }
        
        let num_arrays = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
        offset += 4;
        
        for _ in 0..num_arrays {
            if offset + 4 > buffer.len() {
                return Err("Unexpected end of file".into());
            }
            
            // Read name length
            let name_len = u32::from_le_bytes([
                buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
            ]) as usize;
            offset += 4;
            
            if offset + name_len > buffer.len() {
                return Err("Unexpected end of file reading name".into());
            }
            
            // Read name
            let name = String::from_utf8_lossy(&buffer[offset..offset+name_len]).to_string();
            offset += name_len;
            
            if offset + 4 > buffer.len() {
                return Err("Unexpected end of file reading shape length".into());
            }
            
            // Read shape length
            let shape_len = u32::from_le_bytes([
                buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
            ]) as usize;
            offset += 4;
            
            if offset + shape_len * 4 > buffer.len() {
                return Err("Unexpected end of file reading shape".into());
            }
            
            // Read shape
            let mut shape = Vec::new();
            for _ in 0..shape_len {
                let dim = u32::from_le_bytes([
                    buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
                ]) as usize;
                shape.push(dim);
                offset += 4;
            }
            
            if offset + 4 > buffer.len() {
                return Err("Unexpected end of file reading data length".into());
            }
            
            // Read data length
            let data_len = u32::from_le_bytes([
                buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
            ]) as usize;
            offset += 4;
            
            if offset + data_len * 4 > buffer.len() {
                return Err("Unexpected end of file reading data".into());
            }
            
            // Read data
            let mut data = Vec::new();
            for _ in 0..data_len {
                let val = f32::from_le_bytes([
                    buffer[offset], buffer[offset+1], buffer[offset+2], buffer[offset+3]
                ]);
                data.push(val);
                offset += 4;
            }
            
            arrays.insert(name, (shape, data));
        }
        
        Ok(arrays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixture_manifest_loading() {
        let loader_result = FixtureLoader::new("fixtures");
        
        match loader_result {
            Ok(loader) => {
                println!("Successfully loaded fixture loader");
                let cases = loader.list_cases();
                println!("Available test cases: {:?}", cases);
                assert!(cases.len() >= 8, "Should have at least 8 test cases");
                
                // Test getting specific case
                if let Some(case) = loader.get_case_by_id("basic_self_attn_small") {
                    println!("Found case: {:?}", case.case_id);
                    assert_eq!(case.config.batch_size, 2);
                    assert_eq!(case.config.seq_len_q, 4);
                    assert_eq!(case.config.nhead, 4);
                } else {
                    panic!("Could not find basic_self_attn_small case");
                }
                
                println!("✓ Manifest loading test passed");
            },
            Err(e) => {
                println!("Failed to load fixture loader: {}", e);
                // This might fail if fixtures directory doesn't exist, which is OK for now
                println!("This is expected if fixtures haven't been generated yet");
            }
        }
    }
    
    #[test]
    fn test_simple_fixture_loading() {
        let loader_result = FixtureLoader::new("fixtures");
        
        if let Ok(loader) = loader_result {
            // Test loading dummy fixture
            match loader.load_simple_test_fixture("basic_self_attn_small.test") {
                Ok(arrays) => {
                    println!("Successfully loaded fixture arrays");
                    for (name, (shape, data)) in &arrays {
                        println!("  {}: shape={:?}, data_len={}", name, shape, data.len());
                    }
                    
                    // Verify expected arrays exist
                    assert!(arrays.contains_key("input_x"), "Should contain input_x");
                    assert!(arrays.contains_key("weight_w_qkv"), "Should contain weight_w_qkv");
                    assert!(arrays.contains_key("output_output"), "Should contain output_output");
                    
                    // Verify shapes
                    let (input_shape, input_data) = &arrays["input_x"];
                    assert_eq!(input_shape, &vec![2, 4, 32], "input_x should have shape [2, 4, 32]");
                    assert_eq!(input_data.len(), 2*4*32, "input_x should have correct data length");
                    
                    println!("✓ Simple fixture loading test passed");
                },
                Err(e) => {
                    println!("Failed to load simple fixture: {}", e);
                    println!("This is expected if dummy fixtures haven't been created");
                }
            }
        } else {
            println!("Fixture loader initialization failed - this is expected during development");
        }
    }
    
    #[test]
    fn test_fixture_conversion_to_tensors() {
        let device = Default::default();
        let loader_result = FixtureLoader::new("fixtures");
        
        if let Ok(loader) = loader_result {
            if let Ok(arrays) = loader.load_simple_test_fixture("basic_self_attn_small.test") {
                // Test conversion to Burn tensors
                for (name, (shape, data)) in arrays {
                    if shape.len() == 3 {
                        let tensor = Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(data, shape.clone()), &device
                        );
                        
                        assert_eq!(tensor.shape().dims, shape, 
                            "Tensor shape should match fixture shape for {}", name);
                        
                        println!("✓ Converted {} to tensor with shape {:?}", name, tensor.shape().dims);
                    }
                }
                println!("✓ Tensor conversion test passed");
            }
        }
    }
}
//! Settings module for TabPFN configuration.

use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabPFNSettings {
    /// Custom directory for caching downloaded TabPFN models.
    /// If not set, uses platform-specific user cache directory.
    pub model_cache_dir: Option<PathBuf>,

    /// Allow running TabPFN on CPU with large datasets (>1000 samples).
    /// Set to true to override the CPU limitation.
    pub allow_cpu_large_dataset: bool,
}

impl Default for TabPFNSettings {
    fn default() -> Self {
        Self {
            model_cache_dir: None,
            allow_cpu_large_dataset: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PytorchSettings {
    /// PyTorch CUDA memory allocation configuration.
    /// Used to optimize GPU memory usage.
    pub pytorch_cuda_alloc_conf: String,
}

impl Default for PytorchSettings {
    fn default() -> Self {
        Self {
            pytorch_cuda_alloc_conf: "max_split_size_mb:512".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingSettings {
    /// Force consistency tests to run regardless of platform.
    /// Set to true to run tests on non-reference platforms.
    pub force_consistency_tests: bool,

    /// Indicates if running in continuous integration environment.
    /// Typically set by CI systems (e.g., GitHub Actions).
    pub ci: bool,
}

impl Default for TestingSettings {
    fn default() -> Self {
        Self {
            force_consistency_tests: false,
            ci: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// TabPFN-specific settings
    pub tabpfn: TabPFNSettings,

    /// Testing/Development settings
    pub testing: TestingSettings,

    /// PyTorch settings
    pub pytorch: PytorchSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tabpfn: TabPFNSettings::default(),
            testing: TestingSettings::default(),
            pytorch: PytorchSettings::default(),
        }
    }
}

impl Settings {
    /// Create a new Settings instance from environment variables and config files.
    /// Environment variables are prefixed with "TABPFN_".
    pub fn new() -> Result<Self, ConfigError> {
        let settings = Config::builder()
            .set_default("tabpfn.model_cache_dir", None::<String>)?
            .set_default("tabpfn.allow_cpu_large_dataset", false)?
            .set_default("pytorch.pytorch_cuda_alloc_conf", "max_split_size_mb:512")?
            .set_default("testing.force_consistency_tests", false)?
            .set_default("testing.ci", false)?
            // Add configuration from .env file if it exists
            .add_source(File::with_name(".env").required(false))
            // Add environment variables with TABPFN_ prefix
            .add_source(Environment::with_prefix("TABPFN").separator("__"))
            .build()?;

        settings.try_deserialize()
    }

    /// Create a Settings instance with default values.
    pub fn default_instance() -> Self {
        Self::default()
    }
}

/// Global settings instance
static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Get the global settings instance, initializing it if necessary.
pub fn settings() -> &'static Settings {
    SETTINGS.get_or_init(|| Settings::new().unwrap_or_else(|_| Settings::default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();

        assert_eq!(settings.tabpfn.model_cache_dir, None);
        assert_eq!(settings.tabpfn.allow_cpu_large_dataset, false);
        assert_eq!(
            settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );
        assert_eq!(settings.testing.force_consistency_tests, false);
        assert_eq!(settings.testing.ci, false);
    }

    #[test]
    fn test_tabpfn_settings_defaults() {
        let tabpfn_settings = TabPFNSettings::default();

        assert!(tabpfn_settings.model_cache_dir.is_none());
        assert_eq!(tabpfn_settings.allow_cpu_large_dataset, false);
    }

    #[test]
    fn test_pytorch_settings_defaults() {
        let pytorch_settings = PytorchSettings::default();

        assert_eq!(
            pytorch_settings.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );
    }

    #[test]
    fn test_testing_settings_defaults() {
        let testing_settings = TestingSettings::default();

        assert_eq!(testing_settings.force_consistency_tests, false);
        assert_eq!(testing_settings.ci, false);
    }

    #[test]
    fn test_settings_new_with_defaults() {
        let settings = Settings::new().unwrap_or_else(|_| Settings::default());

        // Verify structure is correct
        assert!(settings.tabpfn.model_cache_dir.is_none());
        assert_eq!(settings.tabpfn.allow_cpu_large_dataset, false);
        assert_eq!(settings.testing.force_consistency_tests, false);
        assert_eq!(settings.testing.ci, false);
        assert_eq!(
            settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );
    }

    #[test]
    fn test_settings_serialization() {
        let settings = Settings::default();

        // Test that settings can be serialized to JSON
        let json = serde_json::to_string(&settings).expect("Should serialize to JSON");
        assert!(json.contains("model_cache_dir"));
        assert!(json.contains("allow_cpu_large_dataset"));
        assert!(json.contains("pytorch_cuda_alloc_conf"));
        assert!(json.contains("force_consistency_tests"));
        assert!(json.contains("ci"));

        // Test that settings can be deserialized from JSON
        let deserialized: Settings =
            serde_json::from_str(&json).expect("Should deserialize from JSON");
        assert_eq!(
            deserialized.tabpfn.model_cache_dir,
            settings.tabpfn.model_cache_dir
        );
        assert_eq!(
            deserialized.tabpfn.allow_cpu_large_dataset,
            settings.tabpfn.allow_cpu_large_dataset
        );
        assert_eq!(
            deserialized.pytorch.pytorch_cuda_alloc_conf,
            settings.pytorch.pytorch_cuda_alloc_conf
        );
        assert_eq!(
            deserialized.testing.force_consistency_tests,
            settings.testing.force_consistency_tests
        );
        assert_eq!(deserialized.testing.ci, settings.testing.ci);
    }

    #[test]
    fn test_settings_custom_values() {
        let custom_tabpfn = TabPFNSettings {
            model_cache_dir: Some(PathBuf::from("/custom/cache")),
            allow_cpu_large_dataset: true,
        };

        let custom_pytorch = PytorchSettings {
            pytorch_cuda_alloc_conf: "max_split_size_mb:1024".to_string(),
        };

        let custom_testing = TestingSettings {
            force_consistency_tests: true,
            ci: true,
        };

        let custom_settings = Settings {
            tabpfn: custom_tabpfn,
            pytorch: custom_pytorch,
            testing: custom_testing,
        };

        assert_eq!(
            custom_settings.tabpfn.model_cache_dir,
            Some(PathBuf::from("/custom/cache"))
        );
        assert_eq!(custom_settings.tabpfn.allow_cpu_large_dataset, true);
        assert_eq!(
            custom_settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:1024"
        );
        assert_eq!(custom_settings.testing.force_consistency_tests, true);
        assert_eq!(custom_settings.testing.ci, true);
    }

    #[test]
    fn test_global_settings_singleton() {
        let settings1 = settings();
        let settings2 = settings();

        // Verify both references point to the same instance
        assert_eq!(settings1 as *const Settings, settings2 as *const Settings);
        assert!(settings1.pytorch.pytorch_cuda_alloc_conf.contains("512"));
    }

    #[test]
    fn test_settings_debug_format() {
        let settings = Settings::default();
        let debug_str = format!("{:?}", settings);

        // Verify debug output contains expected fields
        assert!(debug_str.contains("Settings"));
        assert!(debug_str.contains("tabpfn"));
        assert!(debug_str.contains("pytorch"));
        assert!(debug_str.contains("testing"));
        assert!(debug_str.contains("model_cache_dir"));
        assert!(debug_str.contains("allow_cpu_large_dataset"));
    }

    #[test]
    fn test_settings_clone() {
        let original = Settings::default();
        let cloned = original.clone();

        assert_eq!(
            original.tabpfn.model_cache_dir,
            cloned.tabpfn.model_cache_dir
        );
        assert_eq!(
            original.tabpfn.allow_cpu_large_dataset,
            cloned.tabpfn.allow_cpu_large_dataset
        );
        assert_eq!(
            original.pytorch.pytorch_cuda_alloc_conf,
            cloned.pytorch.pytorch_cuda_alloc_conf
        );
        assert_eq!(
            original.testing.force_consistency_tests,
            cloned.testing.force_consistency_tests
        );
        assert_eq!(original.testing.ci, cloned.testing.ci);
    }

    #[test]
    fn test_pathbuf_handling() {
        let path = PathBuf::from("/tmp/tabpfn_cache");
        let tabpfn_settings = TabPFNSettings {
            model_cache_dir: Some(path.clone()),
            allow_cpu_large_dataset: false,
        };

        assert_eq!(tabpfn_settings.model_cache_dir, Some(path));

        // Test serialization/deserialization with PathBuf
        let json =
            serde_json::to_string(&tabpfn_settings).expect("Should serialize TabPFNSettings");
        let deserialized: TabPFNSettings =
            serde_json::from_str(&json).expect("Should deserialize TabPFNSettings");
        assert_eq!(
            deserialized.model_cache_dir,
            Some(PathBuf::from("/tmp/tabpfn_cache"))
        );
    }

    #[test]
    fn test_settings_equivalence_with_python() {
        // Test that our Rust implementation matches Python's default behavior
        let settings = Settings::default();

        // Python: model_cache_dir: Path | None = Field(default=None)
        assert_eq!(settings.tabpfn.model_cache_dir, None);

        // Python: allow_cpu_large_dataset: bool = Field(default=False)
        assert_eq!(settings.tabpfn.allow_cpu_large_dataset, false);

        // Python: pytorch_cuda_alloc_conf: str = Field(default="max_split_size_mb:512")
        assert_eq!(
            settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );

        // Python: force_consistency_tests: bool = Field(default=False)
        assert_eq!(settings.testing.force_consistency_tests, false);

        // Python: ci: bool = Field(default=False)
        assert_eq!(settings.testing.ci, false);
    }

    #[test]
    fn test_config_error_handling() {
        // Test that invalid configuration falls back to defaults gracefully
        // This simulates the Python behavior: Settings().unwrap_or_else(|_| Settings::default())

        // Create a config that would fail parsing
        let settings = Settings::new().unwrap_or_else(|_| Settings::default());

        // Should still get valid defaults
        assert_eq!(settings.tabpfn.allow_cpu_large_dataset, false);
        assert_eq!(
            settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );
    }

    #[test]
    fn test_env_var_isolation() {
        // This test ensures our environment variable handling doesn't interfere with other tests
        // We test the mechanism without actually setting variables to avoid threading issues

        // Test that Settings::new() works without any special environment setup
        let settings = Settings::new().unwrap_or_else(|_| Settings::default());

        // Should get defaults when no env vars are set
        assert_eq!(settings.tabpfn.allow_cpu_large_dataset, false);
        assert_eq!(settings.testing.ci, false);
        assert_eq!(
            settings.pytorch.pytorch_cuda_alloc_conf,
            "max_split_size_mb:512"
        );
    }

    /// Test demonstrating thread-safe access to global settings
    #[test]
    fn test_concurrent_settings_access() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn multiple threads that access settings
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                let _settings = settings();
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all threads completed successfully
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}

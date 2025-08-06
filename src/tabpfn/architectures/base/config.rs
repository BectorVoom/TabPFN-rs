//  Copyright (c) Prior Labs GmbH 2025.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature positional embedding types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePositionalEmbedding {
    NormalRandVec,
    UniRandVec,
    Learned,
    Subspace,
}

impl Default for FeaturePositionalEmbedding {
    fn default() -> Self {
        Self::Subspace
    }
}

/// Base architecture configuration trait
pub trait ArchitectureConfig {
    fn max_num_classes(&self) -> i32;
    fn num_buckets(&self) -> i32;
    fn get_unused_config(&self, unparsed_config: &HashMap<String, serde_json::Value>) -> HashMap<String, serde_json::Value>;
}

/// Configuration for the base architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Required fields from ArchitectureConfig
    pub max_num_classes: i32,
    pub num_buckets: i32,
    
    // ------ Actual variation across configs
    /// The embedding dimension
    #[serde(default = "default_emsize")]
    pub emsize: i32,
    
    /// If > 1, the features will be grouped into groups of this size and the attention
    /// is across groups
    #[serde(default = "default_features_per_group")]
    pub features_per_group: i32, // 1 or 2
    
    /// Number of attention heads for both between-item and between-feature attention
    #[serde(default = "default_nhead")]
    pub nhead: i32,
    
    #[serde(default)]
    pub remove_duplicate_features: bool,

    // --- Constant across all configs and used
    #[serde(default)]
    pub dropout: f64,
    
    #[serde(default)]
    pub encoder_use_bias: bool,
    
    #[serde(default)]
    pub feature_positional_embedding: Option<FeaturePositionalEmbedding>,
    
    /// When True, uses multiquery for attention between items
    #[serde(default)]
    pub multiquery_item_attention: bool, // Always false
    
    /// NaN handling enabled
    #[serde(default = "default_true")]
    pub nan_handling_enabled: bool,
    
    /// NaN handling for y encoder
    #[serde(default = "default_true")]
    pub nan_handling_y_encoder: bool,
    
    /// Hidden dimension in the MLP layers is ninp * nhid_factor
    #[serde(default = "default_nhid_factor")]
    pub nhid_factor: i32,
    
    /// Number of layers in the encoder, each consisting of
    /// a multi-head attention and an MLP layer
    #[serde(default = "default_nlayers")]
    pub nlayers: i32,
    
    /// Normalize by used features
    #[serde(default = "default_true")]
    pub normalize_by_used_features: bool,
    
    /// Normalize on train only
    #[serde(default = "default_true")]
    pub normalize_on_train_only: bool,
    
    /// Normalize to ranking
    #[serde(default)]
    pub normalize_to_ranking: bool,
    
    /// Normalize x
    #[serde(default = "default_true")]
    pub normalize_x: bool,
    
    /// If True, enables activation checkpointing for each attention layer **and each
    /// MLP layer** in the encoder. This saves memory. recompute_layer is a related flag
    /// which checkpoints the input to each PerFeatureEncoderLayer
    #[serde(default)]
    pub recompute_attn: bool,
    
    /// If True, enables activation checkpointing for each PerFeatureEncoderLayer in the
    /// encoder. This saves memory. recompute_attn is a related flag which checkpoints the
    /// attention and mlp layers individually
    #[serde(default = "default_true")]
    pub recompute_layer: bool,
    
    /// Remove empty features
    #[serde(default = "default_true")]
    pub remove_empty_features: bool,
    
    /// Remove outliers
    #[serde(default)]
    pub remove_outliers: bool,
    
    /// If True, the decoder will be separate from the encoder
    #[serde(default)]
    pub use_separate_decoder: bool,
    
    /// If true, uses multiquery attention on the test set
    #[serde(default = "default_true")]
    pub multiquery_item_attention_for_test_set: bool,
    
    /// The gain when initializing the attention parameters. If None, then 1.0 is used
    #[serde(default = "default_attention_init_gain")]
    pub attention_init_gain: f64,
    

    
    /// DAG positional encoding dimension
    pub dag_pos_enc_dim: Option<i32>,
    
    /// Item attention type
    #[serde(default = "default_full")]
    pub item_attention_type: String,
    
    /// Feature attention type
    #[serde(default = "default_full")]
    pub feature_attention_type: String,
    
    /// The seed to use for the model. The default 0 is chosen to match
    /// the default random_state of 0 in the TabPFN estimator,
    /// which was used to set this seed before
    /// (though I'm not sure it makes a difference for a trained model)
    #[serde(default)]
    pub seed: i32,
}

// Default value functions
fn default_emsize() -> i32 { 192 }
fn default_features_per_group() -> i32 { 2 }
fn default_nhead() -> i32 { 6 }
fn default_true() -> bool { true }
fn default_nhid_factor() -> i32 { 4 }
fn default_nlayers() -> i32 { 12 }
fn default_attention_init_gain() -> f64 { 1.0 }
fn default_full() -> String { "full".to_string() }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_num_classes: 0, // Must be set by user
            num_buckets: 0,     // Must be set by user
            emsize: default_emsize(),
            features_per_group: default_features_per_group(),
            nhead: default_nhead(),
            remove_duplicate_features: false,
            dropout: 0.0,
            encoder_use_bias: false,
            feature_positional_embedding: Some(FeaturePositionalEmbedding::default()),
            multiquery_item_attention: false,
            nan_handling_enabled: true,
            nan_handling_y_encoder: true,
            nhid_factor: default_nhid_factor(),
            nlayers: default_nlayers(),
            normalize_by_used_features: true,
            normalize_on_train_only: true,
            normalize_to_ranking: false,
            normalize_x: true,
            recompute_attn: false,
            recompute_layer: true,
            remove_empty_features: true,
            remove_outliers: false,
            use_separate_decoder: false,
            multiquery_item_attention_for_test_set: true,
            attention_init_gain: default_attention_init_gain(),
            dag_pos_enc_dim: None,
            item_attention_type: default_full(),
            feature_attention_type: default_full(),
            seed: 0,
        }
    }
}

impl ArchitectureConfig for ModelConfig {
    fn max_num_classes(&self) -> i32 {
        self.max_num_classes
    }
    
    fn num_buckets(&self) -> i32 {
        self.num_buckets
    }
    
    fn get_unused_config(&self, unparsed_config: &HashMap<String, serde_json::Value>) -> HashMap<String, serde_json::Value> {
        get_unused_items(unparsed_config, &self.to_dict())
    }
}

impl ModelConfig {
    /// Upgrade old configs to match the current config.
    /// 
    /// This allows backwards compatibility with checkpoints.
    /// Returns an error if the config is not compatible with the current code.
    pub fn upgrade_config(mut config: HashMap<String, serde_json::Value>) -> Result<HashMap<String, serde_json::Value>, String> {
        // The dates are to help us remove upgrades when they get very old.
        
        // Config changed on unknown date
        if config.remove("use_flash_attention").is_some() {
            log::debug!(
                "`use_flash_attention` was specified in the config. This will be \
                ignored and the attention implementation selected automatically."
            );
        }
        
        // Config changed on 2025-05-22
        // Some keys were previously allowed to be None, and replaced with a default
        // value when they were used. Now we keep the default value in the configs and
        // None isn't allowed, so replace None with the default value.
        if let Some(gain) = config.get("attention_init_gain") {
            if gain.is_null() {
                config.insert("attention_init_gain".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(1.0).unwrap()));
            }
        }
        
        // Config changed on 2025-06-03
        if let Some(attention_type) = config.remove("attention_type") {
            if config.contains_key("item_attention_type") || config.contains_key("feature_attention_type") {
                return Err("Can't have both old and new attention types set".to_string());
            }
            config.insert("item_attention_type".to_string(), attention_type.clone());
            config.insert("feature_attention_type".to_string(), attention_type);
        }
        
        // Config changed on 2025-06-04
        if let Some(canonical_y_encoder) = config.get("canonical_y_encoder") {
            if !canonical_y_encoder.is_boolean() || canonical_y_encoder.as_bool().unwrap_or(true) != false {
                return Err("Current version only supports canonical_y_encoder=False".to_string());
            }
        }
        if let Some(bias) = config.get("bias") {
            if !bias.is_boolean() || bias.as_bool().unwrap_or(true) != false {
                return Err("Current version only supports bias=False".to_string());
            }
        }
        
        // Config changed on 2025-07-09
        if let Some(two_sets) = config.remove("two_sets_of_queries") {
            if two_sets.as_bool().unwrap_or(false) {
                return Err("`two_sets_of_queries` is no longer supported in config".to_string());
            }
        }
        
        Ok(config)
    }
    
    /// Convert to dictionary for compatibility checks
    fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        serde_json::to_value(self)
            .unwrap()
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .collect()
    }
    
    /// Validate consistency of configuration
    pub fn validate_consistent(&self) -> Result<(), String> {
        if self.emsize % self.nhead != 0 {
            return Err("emsize must be divisible by nhead".to_string());
        }
        
        // Validate features_per_group is 1 or 2
        if self.features_per_group != 1 && self.features_per_group != 2 {
            return Err("features_per_group must be 1 or 2".to_string());
        }
        
        Ok(())
    }
}

/// Returns items in the given config that were not parsed by this config.
/// 
/// This emulates Pydantic's extra="allow" and __pydantic_extra__ feature, which
/// unfortunately isn't supported for dataclasses.
fn get_unused_items(
    full_config: &HashMap<String, serde_json::Value>,
    used_config: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut unused = HashMap::new();
    
    for (k, v) in full_config.iter() {
        if !used_config.contains_key(k) {
            unused.insert(k.clone(), v.clone());
        } else if let (Some(v_obj), Some(used_obj)) = (v.as_object(), used_config.get(k).and_then(|x| x.as_object())) {
            let v_map: HashMap<String, serde_json::Value> = v_obj.clone().into_iter().collect();
            let used_map: HashMap<String, serde_json::Value> = used_obj.clone().into_iter().collect();
            let subconfig_unused = get_unused_items(&v_map, &used_map);
            if !subconfig_unused.is_empty() {
                unused.insert(k.clone(), serde_json::to_value(subconfig_unused).unwrap());
            }
        }
    }
    
    unused
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.emsize, 192);
        assert_eq!(config.features_per_group, 2);
        assert_eq!(config.nhead, 6);
        assert!(!config.remove_duplicate_features);
        assert_eq!(config.dropout, 0.0);
        assert!(!config.encoder_use_bias);
        assert_eq!(config.feature_positional_embedding, Some(FeaturePositionalEmbedding::Subspace));
        assert!(!config.multiquery_item_attention);
        assert!(config.nan_handling_enabled);
        assert!(config.nan_handling_y_encoder);
        assert_eq!(config.nhid_factor, 4);
        assert_eq!(config.nlayers, 12);
        assert!(config.normalize_by_used_features);
        assert!(config.normalize_on_train_only);
        assert!(!config.normalize_to_ranking);
        assert!(config.normalize_x);
        assert!(!config.recompute_attn);
        assert!(config.recompute_layer);
        assert!(config.remove_empty_features);
        assert!(!config.remove_outliers);
        assert!(!config.use_separate_decoder);
        assert!(config.multiquery_item_attention_for_test_set);
        assert_eq!(config.attention_init_gain, 1.0);
        assert_eq!(config.dag_pos_enc_dim, None);
        assert_eq!(config.item_attention_type, "full");
        assert_eq!(config.feature_attention_type, "full");
        assert_eq!(config.seed, 0);
    }
    
    #[test]
    fn test_validate_consistent() {
        let mut config = ModelConfig::default();
        
        // Valid config
        config.emsize = 192;
        config.nhead = 6;
        assert!(config.validate_consistent().is_ok());
        
        // Invalid emsize/nhead ratio
        config.emsize = 193;
        config.nhead = 6;
        assert!(config.validate_consistent().is_err());
        
        // Invalid features_per_group
        config.emsize = 192;
        config.features_per_group = 3;
        assert!(config.validate_consistent().is_err());
    }
    
    #[test]
    fn test_upgrade_config() {
        let mut config = HashMap::new();
        config.insert("use_flash_attention".to_string(), json!(true));
        config.insert("attention_init_gain".to_string(), json!(null));
        
        let upgraded = ModelConfig::upgrade_config(config).unwrap();
        assert!(!upgraded.contains_key("use_flash_attention"));
        assert_eq!(upgraded.get("attention_init_gain"), Some(&json!(1.0)));
    }
    
    #[test]
    fn test_upgrade_config_attention_type() {
        let mut config = HashMap::new();
        config.insert("attention_type".to_string(), json!("full"));
        
        let upgraded = ModelConfig::upgrade_config(config).unwrap();
        assert!(!upgraded.contains_key("attention_type"));
        assert_eq!(upgraded.get("item_attention_type"), Some(&json!("full")));
        assert_eq!(upgraded.get("feature_attention_type"), Some(&json!("full")));
    }
    
    #[test]
    fn test_upgrade_config_errors() {
        // Test canonical_y_encoder error
        let mut config = HashMap::new();
        config.insert("canonical_y_encoder".to_string(), json!(true));
        assert!(ModelConfig::upgrade_config(config).is_err());
        
        // Test bias error
        let mut config = HashMap::new();
        config.insert("bias".to_string(), json!(true));
        assert!(ModelConfig::upgrade_config(config).is_err());
        
        // Test two_sets_of_queries error
        let mut config = HashMap::new();
        config.insert("two_sets_of_queries".to_string(), json!(true));
        assert!(ModelConfig::upgrade_config(config).is_err());
        
        // Test conflicting attention types
        let mut config = HashMap::new();
        config.insert("attention_type".to_string(), json!("full"));
        config.insert("item_attention_type".to_string(), json!("full"));
        assert!(ModelConfig::upgrade_config(config).is_err());
    }
    
    #[test]
    fn test_serialization() {
        let config = ModelConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(config.emsize, deserialized.emsize);
        assert_eq!(config.nhead, deserialized.nhead);
        assert_eq!(config.feature_positional_embedding, deserialized.feature_positional_embedding);
    }
}
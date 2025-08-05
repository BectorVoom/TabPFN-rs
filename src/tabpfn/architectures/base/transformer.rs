//  Copyright (c) Prior Labs GmbH 2025.

use burn::{
    module::Module,
    nn,
    tensor::{activation, backend::Backend, Tensor},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use petgraph::{Graph, Directed};
use petgraph::graph::NodeIndex;
use nalgebra::DVector;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use super::{
    config::{FeaturePositionalEmbedding, ModelConfig},
    encoders::{InputEncoder, SequentialEncoder, LinearInputEncoderStep, NanHandlingEncoderStep},
    layer::PerFeatureEncoderLayer,
};
// use crate::tabpfn::architectures::interface::Architecture;

// Graph node metadata for DAG operations
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub is_feature: bool,
    pub is_target: bool,
    pub feature_idxs: Vec<usize>,
    pub target_idxs: Vec<usize>,
    pub positional_encoding: Option<DVector<f64>>,
}

impl NodeMetadata {
    pub fn new() -> Self {
        Self {
            is_feature: false,
            is_target: false,
            feature_idxs: Vec::new(),
            target_idxs: Vec::new(),
            positional_encoding: None,
        }
    }

    pub fn with_feature_indices(mut self, indices: Vec<usize>) -> Self {
        self.is_feature = !indices.is_empty();
        self.feature_idxs = indices;
        self
    }

    pub fn with_target_indices(mut self, indices: Vec<usize>) -> Self {
        self.is_target = !indices.is_empty();
        self.target_idxs = indices;
        self
    }
}

// Type alias for our DAG representation
pub type DataDAG = Graph<NodeMetadata, (), Directed>;

// Random number generator state isolation context
#[derive(Debug)]
pub struct TorchRngContext {
    rng: StdRng,
    saved_seed: u64,
}

impl TorchRngContext {
    pub fn new<B: Backend>(seed: u64, _device: &B::Device) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self {
            rng,
            saved_seed: seed,
        }
    }

    pub fn with_isolated_rng<F, R>(&mut self, seed: u64, f: F) -> R
    where
        F: FnOnce(&mut StdRng) -> R,
    {
        // Create a new RNG with the specified seed
        let mut isolated_rng = StdRng::seed_from_u64(seed);
        let result = f(&mut isolated_rng);
        
        // Restore the original RNG state (in a real implementation, 
        // we'd need to save and restore the actual RNG state)
        self.rng = StdRng::seed_from_u64(self.saved_seed);
        result
    }
}

/// Layer stack with optional layer dropout, similar to nn.Sequential
#[derive(Module, Debug)]
pub struct LayerStack<B: Backend> {
    layers: Vec<PerFeatureEncoderLayer<B>>,
    min_num_layers_layer_dropout: usize,
    recompute_each_layer: bool,
}

impl<B: Backend> LayerStack<B> {
    pub fn new(
        layers: Vec<PerFeatureEncoderLayer<B>>,
        recompute_each_layer: bool,
        min_num_layers_layer_dropout: Option<usize>,
    ) -> Self {
        let min_layers = min_num_layers_layer_dropout.unwrap_or(layers.len());
        Self {
            layers,
            min_num_layers_layer_dropout: min_layers,
            recompute_each_layer,
        }
    }

    pub fn of_repeated_layer<F>(
        layer_creator: F,
        num_layers: usize,
        recompute_each_layer: bool,
        min_num_layers_layer_dropout: Option<usize>,
    ) -> Self
    where
        F: Fn() -> PerFeatureEncoderLayer<B>,
    {
        let layers = (0..num_layers).map(|_| layer_creator()).collect();
        Self::new(layers, recompute_each_layer, min_num_layers_layer_dropout)
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        single_eval_pos: Option<usize>,
        cache_trainset_representation: bool,
    ) -> Tensor<B, 4> {
        // Apply layer dropout during training
        let n_layers = if B::ad_enabled() {
            // During training, randomly select number of layers
            let min_layers = self.min_num_layers_layer_dropout;
            let max_layers = self.layers.len();
            if min_layers >= max_layers {
                max_layers
            } else {
                // For simplicity, we'll use all layers in Rust implementation
                // In a full implementation, this would use random sampling
                max_layers
            }
        } else {
            self.layers.len()
        };

        let mut output = x;
        for layer in self.layers.iter().take(n_layers) {
            if self.recompute_each_layer && B::ad_enabled() {
                // In Burn, gradient checkpointing would be handled differently
                // For now, we'll just apply the layer normally
                output = layer.forward(output, single_eval_pos, cache_trainset_representation);
            } else {
                output = layer.forward(output, single_eval_pos, cache_trainset_representation);
            }
        }

        output
    }
}

/// Per-feature transformer model that processes a token per feature and sample
#[derive(Module, Debug)]
pub struct PerFeatureTransformer<B: Backend> {
    // Core neural network components
    encoder: SequentialEncoder<B>,
    y_encoder: SequentialEncoder<B>,
    transformer_encoder: LayerStack<B>,
    transformer_decoder: Option<LayerStack<B>>,
    
    global_att_embeddings_for_compression: Option<nn::Embedding<B>>,
    encoder_compression_layer: Option<LayerStack<B>>,
    
    decoder_linear1: nn::Linear<B>,
    decoder_linear2: nn::Linear<B>,
    
    feature_positional_embedding_embeddings: Option<nn::Linear<B>>,
    
    // Non-module parameters
    #[module(skip)]
    ninp: usize,
    #[module(skip)]
    nhid_factor: usize,
    #[module(skip)]
    features_per_group: usize,
    #[module(skip)]
    cache_trainset_representation: bool,
    #[module(skip)]
    n_out: usize,
    #[module(skip)]
    feature_positional_embedding: Option<FeaturePositionalEmbedding>,
    #[module(skip)]
    dag_pos_enc_dim: Option<usize>,
    #[module(skip)]
    seed: u64,
    
    // Cached state (not part of module)
    #[module(skip)]
    cached_embeddings: Arc<Mutex<Option<Tensor<B, 2>>>>,
    #[module(skip)]
    cached_feature_positional_embeddings: Arc<Mutex<Option<Tensor<B, 2>>>>,
}

impl<B: Backend> PerFeatureTransformer<B> {
    pub fn new(
        config: &ModelConfig,
        n_out: usize,
        activation: &str,
        min_num_layers_layer_dropout: Option<usize>,
        zero_init: bool,
        nlayers_decoder: Option<usize>,
        use_encoder_compression_layer: bool,
        precomputed_kv: Option<Vec<Tensor<B, 3>>>,
        cache_trainset_representation: bool,
        device: &B::Device,
    ) -> Self {
        let ninp = config.emsize as usize;
        let nhid_factor = config.nhid_factor as usize;
        let nhid = ninp * nhid_factor;
        let features_per_group = config.features_per_group as usize;

        // Create default encoders if not provided (matching Python implementation)
        let encoder = SequentialEncoder::new(vec![
            Box::new(LinearInputEncoderStep::new(
                1, // num_features
                ninp,
                false, // replace_nan_by_zero
                true,  // bias
                vec!["main".to_string()],
                vec!["output".to_string()],
                device,
            )),
        ]);

        let y_encoder = SequentialEncoder::new(vec![
            Box::new(NanHandlingEncoderStep::new(true)),
            Box::new(LinearInputEncoderStep::new(
                2, // num_features for y (main + nan_indicators)
                ninp,
                false, // replace_nan_by_zero
                true,  // bias
                vec!["main".to_string(), "nan_indicators".to_string()],
                vec!["output".to_string()],
                device,
            )),
        ]);

        // Create layer creator function equivalent
        let layer_creator = || {
            PerFeatureEncoderLayer::new(
                config,
                nhid,
                activation,
                zero_init,
                precomputed_kv.as_ref().and_then(|kv| kv.first().cloned()),
                device,
            )
        };

        let transformer_encoder = LayerStack::of_repeated_layer(
            layer_creator,
            config.nlayers as usize,
            config.recompute_layer,
            min_num_layers_layer_dropout,
        );

        let transformer_decoder = if config.use_separate_decoder {
            let nlayers_decoder = nlayers_decoder
                .ok_or("nlayers_decoder must be specified if use_separate_decoder is True")
                .unwrap();
            Some(LayerStack::of_repeated_layer(
                layer_creator,
                nlayers_decoder,
                false,
                None,
            ))
        } else {
            None
        };

        let (global_att_embeddings_for_compression, encoder_compression_layer) = 
            if use_encoder_compression_layer {
                assert!(config.use_separate_decoder, "use_encoder_compression_layer requires use_separate_decoder");
                let num_global_att_tokens_for_compression = 512;

                let embeddings = nn::EmbeddingConfig::new(num_global_att_tokens_for_compression, ninp)
                    .init(device);

                let compression_layer = LayerStack::of_repeated_layer(
                    layer_creator,
                    2,
                    false,
                    None,
                );

                (Some(embeddings), Some(compression_layer))
            } else {
                (None, None)
            };

        // Create decoder layers
        let decoder_linear1 = nn::LinearConfig::new(ninp, nhid).init(device);
        let decoder_linear2 = nn::LinearConfig::new(nhid, n_out).init(device);

        // Feature positional embedding setup
        let feature_positional_embedding_embeddings = match &config.feature_positional_embedding {
            Some(FeaturePositionalEmbedding::Learned) => {
                Some(nn::LinearConfig::new(1000, ninp).init(device))
            }
            Some(FeaturePositionalEmbedding::Subspace) => {
                Some(nn::LinearConfig::new(ninp / 4, ninp).init(device))
            }
            _ => None,
        };

        Self {
            encoder,
            y_encoder,
            transformer_encoder,
            transformer_decoder,
            global_att_embeddings_for_compression,
            encoder_compression_layer,
            decoder_linear1,
            decoder_linear2,
            feature_positional_embedding_embeddings,
            ninp,
            nhid_factor,
            features_per_group,
            cache_trainset_representation,
            n_out,
            feature_positional_embedding: config.feature_positional_embedding.clone(),
            dag_pos_enc_dim: config.dag_pos_enc_dim.map(|d| d as usize),
            seed: config.seed as u64,
            cached_embeddings: Arc::new(Mutex::new(None)),
            cached_feature_positional_embeddings: Arc::new(Mutex::new(None)),
        }
    }

    pub fn reset_save_peak_mem_factor(&mut self, factor: Option<usize>) {
        // Set the save_peak_mem_factor for all layers
        // This would need to be implemented in the layer implementation
        for layer in &mut self.transformer_encoder.layers {
            // layer.save_peak_mem_factor = factor;
        }
    }

    pub fn forward(
        &self,
        x: HashMap<String, Tensor<B, 3>>,
        y: Option<HashMap<String, Tensor<B, 3>>>,
        only_return_standard_out: bool,
        categorical_inds: Option<Vec<Vec<usize>>>,
        style: Option<Tensor<B, 2>>,
        data_dags: Option<Vec<DataDAG>>,
    ) -> Result<Tensor<B, 3>, String> {
        assert!(style.is_none(), "Style is not supported yet");

        // Ensure main key exists in inputs
        let x_main = x.get("main").ok_or("Main must be in input keys")?;
        let [seq_len, batch_size, num_features] = x_main.dims();

        let y = y.unwrap_or_else(|| {
            let mut y_map = HashMap::new();
            y_map.insert("main".to_string(), 
                Tensor::zeros([0, batch_size, 1], x_main.device()));
            y_map
        });

        let y_main = y.get("main").ok_or("Main must be in y keys")?;
        let training_targets_provided = y_main.dims()[0] > 0;

        if !training_targets_provided && !self.cache_trainset_representation {
            return Err("If caching the training data representation is disabled, then you must provide some training labels".to_string());
        }

        let single_eval_pos = y_main.dims()[0];

        // Process input tensors - pad to multiple of features_per_group
        let mut x_processed = x;
        for (k, tensor) in x_processed.iter_mut() {
            let [s, b, f] = tensor.dims();
            let missing_to_next = (self.features_per_group - (f % self.features_per_group)) % self.features_per_group;
            
            if missing_to_next > 0 {
                let padding = Tensor::zeros([s, b, missing_to_next], tensor.device());
                *tensor = Tensor::cat(vec![tensor.clone(), padding], 2);
            }
        }

        // Reshape for feature groups
        for (_, tensor) in x_processed.iter_mut() {
            let [s, b, f] = tensor.dims();
            let n = self.features_per_group;
            // Reshape from (s, b, f) to (b, s, f/n, n)
            *tensor = tensor.clone().reshape([b, s, f / n, n]);
        }

        // Process y tensors
        let mut y_processed = y;
        for (_, tensor) in y_processed.iter_mut() {
            let original_dims = tensor.dims();
            
            // Ensure 3D: s b 1
            if original_dims.len() == 1 {
                *tensor = tensor.clone().unsqueeze_dim(1);
            }
            if original_dims.len() == 2 {
                *tensor = tensor.clone().unsqueeze_dim(2);
            }

            // Transpose: s b 1 -> b s 1
            *tensor = tensor.clone().swap_dims(0, 1);

            let [b, s, _] = tensor.dims();
            if s < seq_len {
                assert_eq!(s, single_eval_pos, "For main y, y must not be given for target time steps");
                
                let padding = Tensor::zeros([b, seq_len - s, 1], tensor.device()).mul_scalar(f64::NAN);
                *tensor = Tensor::cat(vec![tensor.clone(), padding], 1);
            }

            // Transpose back: b s 1 -> s b 1
            *tensor = tensor.clone().swap_dims(0, 1);
        }

        // Ensure no label leakage
        let y_main = y_processed.get_mut("main").unwrap();
        let leak_prevention = Tensor::zeros([seq_len - single_eval_pos, batch_size, 1], y_main.device())
            .mul_scalar(f64::NAN);
        // This would need proper slicing implementation in Burn
        // y_main[single_eval_pos..] = leak_prevention;

        // Encode y
        let embedded_y = self.y_encoder.forward(
            &y_processed,
            Some(single_eval_pos),
            self.cache_trainset_representation,
        )?.swap_dims(0, 1);

        // Encode x  
        let embedded_x = self.encoder.forward(
            &x_processed,
            Some(single_eval_pos),
            self.cache_trainset_representation,
        )?;

        let [b, s, f, e] = embedded_x.dims();
        
        // Add embeddings
        let (embedded_x, embedded_y) = self.add_embeddings(
            embedded_x,
            embedded_y,
            data_dags,
            num_features,
            seq_len,
            self.cache_trainset_representation && training_targets_provided,
            self.cache_trainset_representation && !training_targets_provided,
        )?;

        // Combine x and y: b s f e + b s 1 e -> b s f+1 e
        let embedded_input = Tensor::cat(vec![embedded_x, embedded_y.unsqueeze_dim(2)], 2);

        // Apply transformer encoder
        let encoder_input = if self.transformer_decoder.is_some() {
            // Only use training data for encoder if using decoder
            embedded_input.clone().narrow(1, 0, single_eval_pos)
        } else {
            embedded_input.clone()
        };

        let encoder_out = self.transformer_encoder.forward(
            encoder_input,
            Some(single_eval_pos),
            self.cache_trainset_representation,
        );

        let final_encoder_out = if let Some(ref decoder) = self.transformer_decoder {
            // Apply decoder for test data
            let test_input = embedded_input.narrow(1, single_eval_pos, seq_len - single_eval_pos);
            let test_encoder_out = decoder.forward(
                test_input,
                Some(0),
                false,
            );
            Tensor::cat(vec![encoder_out, test_encoder_out], 1)
        } else {
            encoder_out
        };

        // Extract test outputs (last feature dimension, which is the target)
        let test_encoder_out = final_encoder_out
            .narrow(1, single_eval_pos, seq_len - single_eval_pos)
            .narrow(2, f, 1)  // Get the last feature (target)
            .squeeze_dim(2)
            .swap_dims(0, 1);  // s b e

        // Apply decoder
        let hidden = self.decoder_linear1.forward(test_encoder_out);
        let activated = activation::gelu(hidden);
        Ok(self.decoder_linear2.forward(activated))
    }

    fn add_embeddings(
        &self,
        mut x: Tensor<B, 4>,
        y: Tensor<B, 3>,
        data_dags: Option<Vec<DataDAG>>,
        num_features: usize,
        seq_len: usize,
        cache_embeddings: bool,
        use_cached_embeddings: bool,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 3>), String> {
        // Check cached embeddings first
        if use_cached_embeddings {
            if let Ok(cached) = self.cached_embeddings.lock() {
                if let Some(ref embs) = *cached {
                    let unsqueezed_embs = embs.clone().unsqueeze_dims(&[0, 1]);
                    x = x + unsqueezed_embs;
                    return Ok((x, y));
                }
            }
        }
        
        // Early return if DAG processing is needed but embeddings would be cached
        if data_dags.is_some() && cache_embeddings {
            return Err("Caching embeddings is not supported with data_dags at this point".to_string());
        }

        // Generate embeddings based on type
        let embs = match &self.feature_positional_embedding {
            Some(FeaturePositionalEmbedding::NormalRandVec) => {
                Some(Tensor::<B, 2>::random([x.dims()[2], x.dims()[3]], burn::tensor::Distribution::Normal(0.0, 1.0), &x.device()))
            }
            Some(FeaturePositionalEmbedding::UniRandVec) => {
                let uniform: Tensor<B, 2> = Tensor::random([x.dims()[2], x.dims()[3]], burn::tensor::Distribution::Uniform(0.0, 1.0), &x.device());
                Some(uniform * 2.0 - 1.0)
            }
            Some(FeaturePositionalEmbedding::Learned) => {
                if let Some(ref embeddings) = self.feature_positional_embedding_embeddings {
                    // Generate random indices and embed them
                    let indices: Tensor<B, 1> = Tensor::random([x.dims()[2]], burn::tensor::Distribution::Uniform(0.0, 1000.0), &x.device());
                    // This would need proper embedding lookup implementation
                    None // Placeholder
                } else {
                    None
                }
            }
            Some(FeaturePositionalEmbedding::Subspace) => {
                if let Some(ref embeddings) = self.feature_positional_embedding_embeddings {
                    let embs: Tensor<B, 2> = Tensor::random([x.dims()[2], x.dims()[3] / 4], burn::tensor::Distribution::Normal(0.0, 1.0), &x.device());
                    Some(embeddings.forward(embs))
                } else {
                    None
                }
            }
            None => None,
        };

        if let Some(embs) = &embs {
            x = x + embs.clone().unsqueeze_dims(&[0, 1]);
        }

        // Cache embeddings if requested (only if no DAGs)
        if cache_embeddings && data_dags.is_none() {
            if let Ok(mut cached) = self.cached_embeddings.lock() {
                *cached = embs.clone();
            }
        }

        // Handle DAG positional encoding
        if let Some(mut dags) = data_dags {
            if let Some(dag_pos_enc_dim) = self.dag_pos_enc_dim {
                let batch_size = x.dims()[0];
                
                for (b_i, dag) in dags.iter_mut().enumerate() {
                    if b_i >= batch_size {
                        break;
                    }
                    
                    // Create a copy of the DAG to process
                    let mut g = dag.clone();
                    
                    // Add transitive closure connections repeatedly until no more connections are added
                    while networkx_add_direct_connections(&mut g) {
                        // Continue until convergence
                    }
                    
                    // Create subgraph with only feature and target nodes
                    let mut subgraph = DataDAG::new();
                    let mut node_mapping = HashMap::new();
                    
                    // Add nodes that are features or targets
                    for node_id in g.node_indices() {
                        if let Some(node_data) = g.node_weight(node_id) {
                            if node_data.is_feature || node_data.is_target {
                                let new_node = subgraph.add_node(node_data.clone());
                                node_mapping.insert(node_id, new_node);
                            }
                        }
                    }
                    
                    // Add edges between the selected nodes
                    for edge_id in g.edge_indices() {
                        if let Some((source, target)) = g.edge_endpoints(edge_id) {
                            if let (Some(&new_source), Some(&new_target)) = 
                                (node_mapping.get(&source), node_mapping.get(&target)) {
                                subgraph.add_edge(new_source, new_target, ());
                            }
                        }
                    }
                    
                    // Compute positional embeddings
                    if let Err(e) = add_pos_emb(&mut subgraph, false, dag_pos_enc_dim) {
                        return Err(format!("Failed to compute positional embeddings: {}", e));
                    }
                    
                    // Extract positional embeddings for features and targets
                    let mut graph_pos_embs_features = vec![vec![0.0; dag_pos_enc_dim]; num_features];
                    let mut graph_pos_embs_targets = vec![vec![0.0; dag_pos_enc_dim]; 1]; // Assuming 1 target
                    
                    // Apply positional embeddings to features and targets
                    for node_id in subgraph.node_indices() {
                        if let Some(node_data) = subgraph.node_weight(node_id) {
                            if let Some(ref pos_enc) = node_data.positional_encoding {
                                // Apply to feature indices
                                for &feature_idx in &node_data.feature_idxs {
                                    if feature_idx < num_features {
                                        for (i, &val) in pos_enc.iter().enumerate() {
                                            if i < dag_pos_enc_dim {
                                                graph_pos_embs_features[feature_idx][i] = val;
                                            }
                                        }
                                    }
                                }
                                
                                // Apply to target indices  
                                for &target_idx in &node_data.target_idxs {
                                    if target_idx < 1 { // Assuming single target
                                        for (i, &val) in pos_enc.iter().enumerate() {
                                            if i < dag_pos_enc_dim {
                                                graph_pos_embs_targets[target_idx][i] = val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Center the embeddings (subtract mean)
                    let mut feature_mean = vec![0.0; dag_pos_enc_dim];
                    for feature_emb in &graph_pos_embs_features {
                        for (i, &val) in feature_emb.iter().enumerate() {
                            feature_mean[i] += val;
                        }
                    }
                    for val in &mut feature_mean {
                        *val /= num_features as f64;
                    }
                    
                    // Subtract mean from target embeddings
                    for target_emb in &mut graph_pos_embs_targets {
                        for (i, val) in target_emb.iter_mut().enumerate() {
                            *val -= feature_mean[i];
                        }
                    }
                    
                    // Subtract mean from feature embeddings
                    for feature_emb in &mut graph_pos_embs_features {
                        for (i, val) in feature_emb.iter_mut().enumerate() {
                            *val -= feature_mean[i];
                        }
                    }
                    
                    // Apply the positional embeddings to tensors
                    // Note: This is a simplified version - in practice, you'd need proper tensor operations
                    // For now, we'll just add them to the first dag_pos_enc_dim dimensions
                    
                    // Apply to x tensor (features) - this would require proper tensor slicing
                    // x[b_i, :, :, :dag_pos_enc_dim] += graph_pos_embs_features (broadcasted)
                    
                    // Apply to y tensor (targets) - this would require proper tensor slicing  
                    // y[b_i, :, :dag_pos_enc_dim] += graph_pos_embs_targets (broadcasted)
                    
                    // TODO: Implement actual tensor operations when Burn supports advanced slicing
                }
            } else {
                return Err("dag_pos_enc_dim must be set when using data_dags".to_string());
            }
        } else {
            // Ensure dag_pos_enc_dim is None or 0 when not using DAGs
            if let Some(dim) = self.dag_pos_enc_dim {
                if dim != 0 {
                    return Err("dag_pos_enc_dim should be None or 0 when not using data_dags".to_string());
                }
            }
        }

        Ok((x, y))
    }

    pub fn empty_trainset_representation_cache(&self) {
        // Clear any cached training data representations
        for layer in &self.transformer_encoder.layers {
            // layer.empty_trainset_representation_cache();
        }
        if let Some(ref decoder) = self.transformer_decoder {
            for layer in &decoder.layers {
                // layer.empty_trainset_representation_cache();
            }
        }
    }
}

// Architecture trait implementation would go here

// Utility functions for networkx graph operations
pub fn networkx_add_direct_connections(graph: &mut DataDAG) -> bool {
    let mut added_connection = false;
    
    // Get all node indices to avoid borrowing issues during iteration
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    
    // Build neighbor maps to avoid repeated computation
    let mut neighbor_map: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for node in &node_indices {
        let neighbors: Vec<NodeIndex> = graph.neighbors(*node).collect();
        neighbor_map.insert(*node, neighbors);
    }
    
    // Iterate over each node
    for node in &node_indices {
        if let Some(neighbors) = neighbor_map.get(node) {
            // Iterate over the neighbors of the current node
            for neighbor in neighbors {
                if let Some(second_neighbors) = neighbor_map.get(neighbor) {
                    // Iterate over the neighbors of the neighbor
                    for second_neighbor in second_neighbors {
                        // Check if a direct edge already exists
                        if !graph.find_edge(*node, *second_neighbor).is_some() {
                            // Add a direct edge from the current node to the second neighbor
                            graph.add_edge(*node, *second_neighbor, ());
                            added_connection = true;
                        }
                    }
                }
            }
        }
    }
    
    added_connection
}

pub fn add_pos_emb(
    graph: &mut DataDAG,
    is_undirected: bool,
    k: usize,
) -> Result<(), String> {
    use nalgebra::{DMatrix, SymmetricEigen};
    
    let node_count = graph.node_count();
    if node_count == 0 {
        return Ok(());
    }
    
    // Create the directed Laplacian matrix
    let mut laplacian = DMatrix::<f64>::zeros(node_count, node_count);
    
    // Create a mapping from NodeIndex to matrix index
    let node_indices: Vec<NodeIndex> = graph.node_indices().collect();
    let mut node_to_idx = HashMap::new();
    for (idx, &node_id) in node_indices.iter().enumerate() {
        node_to_idx.insert(node_id, idx);
    }
    
    // Build the adjacency and degree matrices
    for (i, &node_i) in node_indices.iter().enumerate() {
        let mut out_degree = 0;
        
        // Count outgoing edges and fill adjacency
        for neighbor in graph.neighbors(node_i) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                laplacian[(i, j)] = -1.0;
                out_degree += 1;
            }
        }
        
        // Set diagonal (degree matrix)
        laplacian[(i, i)] = out_degree as f64;
    }
    
    // Handle NaN values (replace with 0.0)
    for i in 0..node_count {
        for j in 0..node_count {
            if laplacian[(i, j)].is_nan() {
                laplacian[(i, j)] = 0.0;
            }
        }
    }
    
    // For directed graphs, we need to use a more complex approach
    // For now, we'll compute the symmetric part for eigendecomposition
    let symmetric_laplacian = if is_undirected {
        laplacian.clone()
    } else {
        // Make it symmetric: (L + L^T) / 2
        let laplacian_t = laplacian.transpose();
        (&laplacian + &laplacian_t) * 0.5
    };
    
    // Compute eigendecomposition
    let eigen = SymmetricEigen::new(symmetric_laplacian);
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;
    
    // Sort by eigenvalues (smallest first for "SR" - smallest real part)
    let mut eigen_pairs: Vec<(f64, DVector<f64>)> = eigenvalues.iter()
        .zip(eigenvectors.column_iter())
        .map(|(&val, vec)| (val, vec.into_owned()))
        .collect();
    
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Take k+1 smallest eigenvalues (excluding the first one)
    let num_eigenvectors = std::cmp::min(k + 1, eigen_pairs.len());
    let start_idx = if eigen_pairs.len() > 1 { 1 } else { 0 }; // Skip first eigenvalue
    let end_idx = std::cmp::min(start_idx + k, eigen_pairs.len());
    
    if start_idx >= end_idx {
        return Ok(());
    }
    
    // Create positional encoding matrix
    let mut pe_matrix = DMatrix::<f64>::zeros(node_count, k);
    for (col_idx, eigen_idx) in (start_idx..end_idx).enumerate() {
        if col_idx >= k { break; }
        let eigenvector = &eigen_pairs[eigen_idx].1;
        for row_idx in 0..node_count {
            pe_matrix[(row_idx, col_idx)] = eigenvector[row_idx];
        }
    }
    
    // Apply random sign flipping
    let mut rng = StdRng::from_entropy();
    for col in 0..k {
        let sign = if rng.r#gen::<bool>() { 1.0 } else { -1.0 };
        for row in 0..node_count {
            pe_matrix[(row, col)] *= sign;
        }
    }
    
    // Assign positional encodings to graph nodes
    for (matrix_idx, &node_id) in node_indices.iter().enumerate() {
        let pe_vector = pe_matrix.row(matrix_idx).transpose().into_owned();
        if let Some(node_weight) = graph.node_weight_mut(node_id) {
            node_weight.positional_encoding = Some(pe_vector);
        }
    }
    
    Ok(())
}
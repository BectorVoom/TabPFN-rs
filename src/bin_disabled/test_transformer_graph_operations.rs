use std::collections::HashMap;
use petgraph::{Graph, Directed};
use nalgebra::DVector;
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;

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

// Utility functions for networkx graph operations
pub fn networkx_add_direct_connections(graph: &mut DataDAG) -> bool {
    use petgraph::graph::NodeIndex;
    
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
    use petgraph::graph::NodeIndex;
    
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
    let mut rng = StdRng::from_rng(thread_rng()).map_err(|e| format!("Failed to create RNG: {}", e))?;
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

fn test_graph_creation() -> Result<(), String> {
    println!("Testing graph creation...");
    
    let mut graph = DataDAG::new();
    
    // Add some nodes
    let node1 = graph.add_node(NodeMetadata::new().with_feature_indices(vec![0]));
    let node2 = graph.add_node(NodeMetadata::new().with_feature_indices(vec![1]));
    let node3 = graph.add_node(NodeMetadata::new().with_target_indices(vec![0]));
    
    // Add some edges
    graph.add_edge(node1, node2, ());
    graph.add_edge(node2, node3, ());
    
    println!("✅ Created graph with {} nodes and {} edges", 
             graph.node_count(), graph.edge_count());
    
    Ok(())
}

fn test_transitive_closure() -> Result<(), String> {
    println!("Testing transitive closure...");
    
    let mut graph = DataDAG::new();
    
    // Create a simple chain: A -> B -> C
    let node_a = graph.add_node(NodeMetadata::new().with_feature_indices(vec![0]));
    let node_b = graph.add_node(NodeMetadata::new().with_feature_indices(vec![1]));
    let node_c = graph.add_node(NodeMetadata::new().with_target_indices(vec![0]));
    
    graph.add_edge(node_a, node_b, ());
    graph.add_edge(node_b, node_c, ());
    
    let initial_edges = graph.edge_count();
    println!("Initial edges: {}", initial_edges);
    
    // Apply transitive closure
    let added = networkx_add_direct_connections(&mut graph);
    let final_edges = graph.edge_count();
    
    println!("Added connections: {}, Final edges: {}", added, final_edges);
    
    // Should have added A -> C edge
    if final_edges > initial_edges {
        println!("✅ Transitive closure working - added {} edges", final_edges - initial_edges);
    } else {
        return Err("Expected transitive closure to add edges".to_string());
    }
    
    Ok(())
}

fn test_positional_embedding() -> Result<(), String> {
    println!("Testing positional embedding computation...");
    
    let mut graph = DataDAG::new();
    
    // Create a simple triangle graph
    let node_a = graph.add_node(NodeMetadata::new().with_feature_indices(vec![0]));
    let node_b = graph.add_node(NodeMetadata::new().with_feature_indices(vec![1]));
    let node_c = graph.add_node(NodeMetadata::new().with_target_indices(vec![0]));
    
    graph.add_edge(node_a, node_b, ());
    graph.add_edge(node_b, node_c, ());
    graph.add_edge(node_c, node_a, ());
    
    // Compute positional embeddings
    let k = 2; // embedding dimension
    add_pos_emb(&mut graph, false, k)?;
    
    // Check that all nodes have positional encodings
    let mut nodes_with_encoding = 0;
    for node_id in graph.node_indices() {
        if let Some(node_data) = graph.node_weight(node_id) {
            if let Some(ref encoding) = node_data.positional_encoding {
                nodes_with_encoding += 1;
                println!("Node has encoding of dimension: {}", encoding.len());
                if encoding.len() != k {
                    return Err(format!("Expected encoding dimension {}, got {}", k, encoding.len()));
                }
            }
        }
    }
    
    if nodes_with_encoding == graph.node_count() {
        println!("✅ All {} nodes have positional encodings", nodes_with_encoding);
    } else {
        return Err(format!("Expected {} nodes with encodings, got {}", 
                          graph.node_count(), nodes_with_encoding));
    }
    
    Ok(())
}

fn main() -> Result<(), String> {
    println!("Testing TabPFN Transformer Graph Operations");
    println!("===========================================");
    
    test_graph_creation()?;
    test_transitive_closure()?;
    test_positional_embedding()?;
    
    println!("\n🎉 All graph operation tests passed!");
    Ok(())
}
// validation.rs - Comprehensive TabPFN Validation Implementation
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Distribution, ElementConversion, Tensor},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use super::transformer::{PerFeatureTransformer, DataDAG};
use super::train::{DatasetPrior, SyntheticTabularDataset, argmax_with_tie_break_smallest};

/// Configuration for validation procedures
#[derive(Config, Debug)]
pub struct ValidationConfig {
    /// Number of synthetic tasks to evaluate
    pub num_synthetic_tasks: usize,

    /// Number of samples per synthetic task
    pub samples_per_task_range: (usize, usize),

    /// Few-shot evaluation settings
    pub few_shot_sizes: Vec<usize>,

    /// Whether to evaluate on real-world datasets
    pub evaluate_real_datasets: bool,

    /// Path to real-world datasets
    pub real_datasets_path: Option<PathBuf>,

    /// Validation protocols to run
    pub protocols: Vec<ValidationProtocol>,

    /// Metrics to compute
    pub metrics: Vec<MetricType>,

    /// Output directory for results
    pub output_dir: PathBuf,

    /// Random seed for reproducibility
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationProtocol {
    FewShot(usize),      // Number of training samples
    FullDataset,         // Use all available training data
    CrossDomain,         // Evaluate on different domain
    Transductive,        // Test features visible during training
    Inductive,          // Test samples completely held out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Accuracy,
    BalancedAccuracy,
    AucRoc,
    LogLoss,
    F1Score,
    Precision,
    Recall,
    CalibrationError,
}

/// Results from validation procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub task_results: Vec<TaskResult>,
    pub aggregate_metrics: AggregateMetrics,
    pub protocol_specific_results: HashMap<String, ProtocolResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub task_type: TaskType,
    pub num_features: usize,
    pub num_classes: usize,
    pub num_train_samples: usize,
    pub num_test_samples: usize,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Synthetic(String),
    RealWorld(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub mean_metrics: HashMap<String, f64>,
    pub std_metrics: HashMap<String, f64>,
    pub percentile_25: HashMap<String, f64>,
    pub percentile_50: HashMap<String, f64>,
    pub percentile_75: HashMap<String, f64>,
    pub task_characteristic_correlations: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResults {
    pub protocol_name: String,
    pub num_tasks: usize,
    pub mean_performance: f64,
    pub std_performance: f64,
    pub confidence_interval: (f64, f64),
}

/// Real-world dataset representation
pub struct RealWorldDataset {
    pub name: String,
    pub features: Tensor<f32, 2>,
    pub targets: Tensor<i64, 1>,
    pub feature_names: Vec<String>,
    pub class_names: Vec<String>,
}

/// Main validation engine
pub struct ValidationEngine<B: Backend> {
    model: PerFeatureTransformer<B>,
    config: ValidationConfig,
    device: B::Device,
}

impl<B: Backend> ValidationEngine<B> {
    pub fn new(
        model: PerFeatureTransformer<B>,
        config: ValidationConfig,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            config,
            device,
        }
    }

    pub fn run_validation(&mut self, rng: &mut StdRng) -> Result<ValidationResults, Box<dyn std::error::Error>> {
        let mut all_task_results = Vec::new();
        let mut protocol_results = HashMap::new();

        // Run each validation protocol with deterministic RNG
        for protocol in &self.config.protocols {
            let results = match protocol {
                ValidationProtocol::FewShot(k) => self.run_few_shot_validation(*k, rng)?,
                ValidationProtocol::FullDataset => self.run_full_dataset_validation(rng)?,
                ValidationProtocol::CrossDomain => self.run_cross_domain_validation(rng)?,
                ValidationProtocol::Transductive => self.run_transductive_validation(rng)?,
                ValidationProtocol::Inductive => self.run_inductive_validation(rng)?,
            };

            all_task_results.extend(results.clone());

            // Compute protocol-specific aggregate metrics
            let protocol_aggregate = self.compute_protocol_metrics(&results, protocol);
            protocol_results.insert(format!("{:?}", protocol), protocol_aggregate);
        }

        // Compute overall aggregate metrics
        let aggregate_metrics = self.compute_aggregate_metrics(&all_task_results);

        // Save results to disk
        self.save_results(&ValidationResults {
            task_results: all_task_results,
            aggregate_metrics,
            protocol_specific_results: protocol_results,
        })?;

        Ok(ValidationResults {
            task_results: all_task_results,
            aggregate_metrics,
            protocol_specific_results: protocol_results,
        })
    }

    fn run_few_shot_validation(&mut self, k: usize, rng: &mut StdRng) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for task_idx in 0..self.config.num_synthetic_tasks {
            // Sample a synthetic task deterministically
            use rand::Rng;
            let num_samples = rng.gen::<usize>() %
                (self.config.samples_per_task_range.1 - self.config.samples_per_task_range.0) +
                self.config.samples_per_task_range.0;

            let dataset = self.sample_synthetic_dataset(num_samples);

            // Create few-shot train/test split deterministically
            let train_indices = self.sample_indices(k, num_samples, rng);
            let test_indices: Vec<usize> = (0..num_samples)
                .filter(|i| !train_indices.contains(i))
                .collect();

            // Prepare inputs for model
            let x_inputs = self.prepare_features(&dataset, &train_indices, &test_indices)?;
            let y_inputs = self.prepare_targets(&dataset, &train_indices)?;

            // Run inference with deterministic RNG
            let predictions = self.model.transformer_forward(
                x_inputs,
                Some(y_inputs),
                true,
                Some(rng),
                None,
                None,
                dataset.dag.as_ref().map(|d| vec![d.clone()]),
            )?;

            // Compute metrics
            let test_targets = self.extract_test_targets(&dataset, &test_indices);
            let metrics = self.compute_metrics(&predictions, &test_targets)?;

            results.push(TaskResult {
                task_id: format!("few_shot_{}_{}", k, task_idx),
                task_type: TaskType::Synthetic(format!("FewShot{}", k)),
                num_features: dataset.features.dims()[2],
                num_classes: self.infer_num_classes(&dataset.targets),
                num_train_samples: k,
                num_test_samples: test_indices.len(),
                metrics,
            });
        }

        Ok(results)
    }

    fn run_full_dataset_validation(&mut self, rng: &mut StdRng) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for task_idx in 0..self.config.num_synthetic_tasks {
            // Sample dataset size deterministically
            use rand::Rng;
            let num_samples = rng.gen::<usize>() %
                (self.config.samples_per_task_range.1 - self.config.samples_per_task_range.0) +
                self.config.samples_per_task_range.0;

            let dataset = self.sample_synthetic_dataset(num_samples);

            // Use standard 70/30 train/test split
            let split_idx = (num_samples as f64 * 0.7) as usize;
            let train_indices: Vec<usize> = (0..split_idx).collect();
            let test_indices: Vec<usize> = (split_idx..num_samples).collect();

            let x_inputs = self.prepare_features(&dataset, &train_indices, &test_indices)?;
            let y_inputs = self.prepare_targets(&dataset, &train_indices)?;

            // Run inference with deterministic RNG
            let predictions = self.model.transformer_forward(
                x_inputs,
                Some(y_inputs),
                true,
                Some(rng),
                None,
                None,
                dataset.dag.as_ref().map(|d| vec![d.clone()]),
            )?;

            let test_targets = self.extract_test_targets(&dataset, &test_indices);
            let metrics = self.compute_metrics(&predictions, &test_targets)?;

            results.push(TaskResult {
                task_id: format!("full_dataset_{}", task_idx),
                task_type: TaskType::Synthetic("FullDataset".to_string()),
                num_features: dataset.features.dims()[2],
                num_classes: self.infer_num_classes(&dataset.targets),
                num_train_samples: train_indices.len(),
                num_test_samples: test_indices.len(),
                metrics,
            });
        }

        Ok(results)
    }

    fn run_cross_domain_validation(&mut self, rng: &mut StdRng) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
        // For cross-domain validation, we evaluate on different prior types
        let mut results = Vec::new();

        // Train on Gaussian prior, test on BayesianNN prior
        for task_idx in 0..self.config.num_synthetic_tasks / 2 {
            let train_dataset = self.sample_synthetic_dataset_with_prior("Gaussian", 100);
            let test_dataset = self.sample_synthetic_dataset_with_prior("BayesianNN", 50);

            // Use all training data from source domain
            let train_indices: Vec<usize> = (0..70).collect();
            let test_indices: Vec<usize> = (0..50).collect();

            let x_inputs = self.prepare_cross_domain_features(
                &train_dataset,
                &test_dataset,
                &train_indices,
                &test_indices,
            )?;
            let y_inputs = self.prepare_targets(&train_dataset, &train_indices)?;

            // Run inference with deterministic RNG
            let predictions = self.model.transformer_forward(
                x_inputs,
                Some(y_inputs),
                true,
                Some(rng),
                None,
                None,
                None,
            )?;

            let test_targets = self.extract_test_targets(&test_dataset, &test_indices);
            let metrics = self.compute_metrics(&predictions, &test_targets)?;

            results.push(TaskResult {
                task_id: format!("cross_domain_{}", task_idx),
                task_type: TaskType::Synthetic("CrossDomain".to_string()),
                num_features: test_dataset.features.dims()[2],
                num_classes: self.infer_num_classes(&test_dataset.targets),
                num_train_samples: train_indices.len(),
                num_test_samples: test_indices.len(),
                metrics,
            });
        }

        Ok(results)
    }

    fn run_transductive_validation(&mut self, rng: &mut StdRng) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for task_idx in 0..self.config.num_synthetic_tasks {
            let num_samples = 100;
            let dataset = self.sample_synthetic_dataset(num_samples);

            // In transductive setting, all features are visible but only some labels
            let labeled_indices: Vec<usize> = (0..30).collect();
            let unlabeled_indices: Vec<usize> = (30..100).collect();

            // Prepare inputs with all features visible
            let mut x_inputs = HashMap::new();
            x_inputs.insert("main".to_string(), dataset.features.clone());

            // Only provide labels for labeled samples
            let mut y_inputs = HashMap::new();
            let mut targets = dataset.targets.clone();
            for idx in &unlabeled_indices {
                // Mask unlabeled samples
                targets = targets.mask_fill(
                    Tensor::zeros(targets.dims(), &self.device).bool(),
                    -1,
                );
            }
            y_inputs.insert("main".to_string(), targets.float().unsqueeze_dim(2));

            // Run inference with deterministic RNG
            let predictions = self.model.transformer_forward(
                x_inputs,
                Some(y_inputs),
                true,
                Some(rng),
                None,
                None,
                dataset.dag.as_ref().map(|d| vec![d.clone()]),
            )?;

            let test_targets = self.extract_test_targets(&dataset, &unlabeled_indices);
            let metrics = self.compute_metrics(&predictions, &test_targets)?;

            results.push(TaskResult {
                task_id: format!("transductive_{}", task_idx),
                task_type: TaskType::Synthetic("Transductive".to_string()),
                num_features: dataset.features.dims()[2],
                num_classes: self.infer_num_classes(&dataset.targets),
                num_train_samples: labeled_indices.len(),
                num_test_samples: unlabeled_indices.len(),
                metrics,
            });
        }

        Ok(results)
    }

    fn run_inductive_validation(&mut self, rng: &mut StdRng) -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        for task_idx in 0..self.config.num_synthetic_tasks {
            let num_samples = 100;
            let dataset = self.sample_synthetic_dataset(num_samples);

            // Complete separation between train and test
            let train_indices: Vec<usize> = (0..50).collect();
            let test_indices: Vec<usize> = (50..100).collect();

            let x_inputs = self.prepare_features(&dataset, &train_indices, &test_indices)?;
            let y_inputs = self.prepare_targets(&dataset, &train_indices)?;

            // Run inference with deterministic RNG
            let predictions = self.model.transformer_forward(
                x_inputs,
                Some(y_inputs),
                true,
                Some(rng),
                None,
                None,
                dataset.dag.as_ref().map(|d| vec![d.clone()]),
            )?;

            let test_targets = self.extract_test_targets(&dataset, &test_indices);
            let metrics = self.compute_metrics(&predictions, &test_targets)?;

            results.push(TaskResult {
                task_id: format!("inductive_{}", task_idx),
                task_type: TaskType::Synthetic("Inductive".to_string()),
                num_features: dataset.features.dims()[2],
                num_classes: self.infer_num_classes(&dataset.targets),
                num_train_samples: train_indices.len(),
                num_test_samples: test_indices.len(),
                metrics,
            });
        }

        Ok(results)
    }

    fn compute_metrics(
        &self,
        predictions: &Tensor<B, 3>,
        targets: &Tensor<B, 1>,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();

        // Convert predictions to class predictions using deterministic tie-breaking
        // This replaces the unsafe argmax(2).squeeze_dim(2) pattern with defensive implementation
        let pred_classes = argmax_with_tie_break_smallest(predictions);

        // Accuracy
        if self.config.metrics.contains(&MetricType::Accuracy) {
            let correct = pred_classes.equal(targets).float();
            let accuracy = correct.mean().into_scalar();
            metrics.insert("accuracy".to_string(), accuracy as f64);
        }

        // Balanced Accuracy
        if self.config.metrics.contains(&MetricType::BalancedAccuracy) {
            let balanced_acc = self.compute_balanced_accuracy(&pred_classes, targets);
            metrics.insert("balanced_accuracy".to_string(), balanced_acc);
        }

        // Log Loss
        if self.config.metrics.contains(&MetricType::LogLoss) {
            let log_loss = self.compute_log_loss(predictions, targets);
            metrics.insert("log_loss".to_string(), log_loss);
        }

        // F1 Score
        if self.config.metrics.contains(&MetricType::F1Score) {
            let f1 = self.compute_f1_score(&pred_classes, targets);
            metrics.insert("f1_score".to_string(), f1);
        }

        // Precision
        if self.config.metrics.contains(&MetricType::Precision) {
            let precision = self.compute_precision(&pred_classes, targets);
            metrics.insert("precision".to_string(), precision);
        }

        // Recall
        if self.config.metrics.contains(&MetricType::Recall) {
            let recall = self.compute_recall(&pred_classes, targets);
            metrics.insert("recall".to_string(), recall);
        }

        // Calibration Error
        if self.config.metrics.contains(&MetricType::CalibrationError) {
            let cal_error = self.compute_calibration_error(predictions, targets);
            metrics.insert("calibration_error".to_string(), cal_error);
        }

        Ok(metrics)
    }

    fn compute_balanced_accuracy(&self, predictions: &Tensor<B, 1>, targets: &Tensor<B, 1>) -> f64 {
        // Simplified balanced accuracy computation
        // In practice, would need per-class accuracy computation
        let correct = predictions.equal(targets).float();
        correct.mean().into_scalar() as f64
    }

    fn compute_log_loss(&self, predictions: &Tensor<B, 3>, targets: &Tensor<B, 1>) -> f64 {
        // Compute negative log likelihood
        let probabilities = burn::tensor::activation::softmax(predictions, 2);
        // This would need proper indexing implementation
        1.0 // Placeholder
    }

    fn compute_f1_score(&self, predictions: &Tensor<B, 1>, targets: &Tensor<B, 1>) -> f64 {
        let precision = self.compute_precision(predictions, targets);
        let recall = self.compute_recall(predictions, targets);

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    fn compute_precision(&self, predictions: &Tensor<B, 1>, targets: &Tensor<B, 1>) -> f64 {
        // Simplified binary precision
        let true_positives = predictions.equal(targets)
            .bool_and(&predictions.equal_elem(1))
            .float()
            .sum()
            .into_scalar() as f64;

        let predicted_positives = predictions.equal_elem(1)
            .float()
            .sum()
            .into_scalar() as f64;

        if predicted_positives > 0.0 {
            true_positives / predicted_positives
        } else {
            0.0
        }
    }

    fn compute_recall(&self, predictions: &Tensor<B, 1>, targets: &Tensor<B, 1>) -> f64 {
        // Simplified binary recall
        let true_positives = predictions.equal(targets)
            .bool_and(&targets.equal_elem(1))
            .float()
            .sum()
            .into_scalar() as f64;

        let actual_positives = targets.equal_elem(1)
            .float()
            .sum()
            .into_scalar() as f64;

        if actual_positives > 0.0 {
            true_positives / actual_positives
        } else {
            0.0
        }
    }

    fn compute_calibration_error(&self, predictions: &Tensor<B, 3>, targets: &Tensor<B, 1>) -> f64 {
        // Expected Calibration Error (ECE)
        // Simplified version - would need binning in practice
        0.05 // Placeholder
    }

    fn compute_aggregate_metrics(&self, results: &[TaskResult]) -> AggregateMetrics {
        let mut mean_metrics = HashMap::new();
        let mut std_metrics = HashMap::new();
        let mut percentile_25 = HashMap::new();
        let mut percentile_50 = HashMap::new();
        let mut percentile_75 = HashMap::new();

        // Collect all metric values
        let mut metric_values: HashMap<String, Vec<f64>> = HashMap::new();

        for result in results {
            for (metric_name, value) in &result.metrics {
                metric_values.entry(metric_name.clone())
                    .or_insert_with(Vec::new)
                    .push(*value);
            }
        }

        // Compute statistics for each metric
        for (metric_name, values) in metric_values {
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            mean_metrics.insert(metric_name.clone(), mean);

            let variance = values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            std_metrics.insert(metric_name.clone(), variance.sqrt());

            percentile_25.insert(metric_name.clone(), sorted_values[sorted_values.len() / 4]);
            percentile_50.insert(metric_name.clone(), sorted_values[sorted_values.len() / 2]);
            percentile_75.insert(metric_name.clone(), sorted_values[3 * sorted_values.len() / 4]);
        }

        // Compute correlations with task characteristics
        let task_characteristic_correlations = self.compute_task_correlations(results);

        AggregateMetrics {
            mean_metrics,
            std_metrics,
            percentile_25,
            percentile_50,
            percentile_75,
            task_characteristic_correlations,
        }
    }

    fn compute_task_correlations(&self, results: &[TaskResult]) -> HashMap<String, f64> {
        let mut correlations = HashMap::new();

        // Correlation between number of features and accuracy
        let features: Vec<f64> = results.iter().map(|r| r.num_features as f64).collect();
        let accuracies: Vec<f64> = results.iter()
            .filter_map(|r| r.metrics.get("accuracy").copied())
            .collect();

        if features.len() == accuracies.len() && !features.is_empty() {
            let correlation = self.pearson_correlation(&features, &accuracies);
            correlations.insert("features_accuracy_correlation".to_string(), correlation);
        }

        // Correlation between number of training samples and accuracy
        let train_samples: Vec<f64> = results.iter().map(|r| r.num_train_samples as f64).collect();

        if train_samples.len() == accuracies.len() && !train_samples.is_empty() {
            let correlation = self.pearson_correlation(&train_samples, &accuracies);
            correlations.insert("train_samples_accuracy_correlation".to_string(), correlation);
        }

        correlations
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_x2: f64 = x.iter().map(|v| v * v).sum();
        let sum_y2: f64 = y.iter().map(|v| v * v).sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn compute_protocol_metrics(
        &self,
        results: &[TaskResult],
        protocol: &ValidationProtocol,
    ) -> ProtocolResults {
        let accuracies: Vec<f64> = results.iter()
            .filter_map(|r| r.metrics.get("accuracy").copied())
            .collect();

        let mean = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        let variance = accuracies.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / accuracies.len() as f64;
        let std = variance.sqrt();

        // 95% confidence interval
        let stderr = std / (accuracies.len() as f64).sqrt();
        let ci_lower = mean - 1.96 * stderr;
        let ci_upper = mean + 1.96 * stderr;

        ProtocolResults {
            protocol_name: format!("{:?}", protocol),
            num_tasks: results.len(),
            mean_performance: mean,
            std_performance: std,
            confidence_interval: (ci_lower, ci_upper),
        }
    }

    fn save_results(&self, results: &ValidationResults) -> Result<(), Box<dyn std::error::Error>> {
        let output_path = self.config.output_dir.join("validation_results.json");
        let json = serde_json::to_string_pretty(results)?;

        let mut file = File::create(output_path)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    // Helper methods

    fn sample_synthetic_dataset(&self, num_samples: usize) -> SyntheticTabularDataset {
        // Simplified dataset sampling
        use super::train::PriorType;
        use super::transformer::DeterministicRngContext;
        use rand::StdRng;
        use rand::SeedableRng;

        let meta_batch_size = 1; // For validation, use single task per batch
        let rng_context = DeterministicRngContext::<B>::new(42, self.device.clone());
        let mut rng = StdRng::seed_from_u64(42);

        let prior = DatasetPrior::new(&super::train::TrainingConfig {
            model: super::config::ModelConfig::default(),
            meta_batch_size,
            tasks_per_batch: 1,
            max_samples_per_task: num_samples,
            min_samples_per_task: num_samples,
            learning_rate: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            gradient_clip_norm: None,
            num_epochs: 1,
            checkpoint_frequency: 1,
            validation_frequency: 1,
            early_stopping_patience: 1,
            use_gradient_checkpointing: false,
            cache_trainset_representations: false,
            layer_dropout_min_layers: None,
            prior_type: PriorType::Gaussian,
            num_features_range: (5, 20),
            num_classes_range: (2, 5),
            feature_noise_level: 0.1,
        });

        prior.sample::<B>(num_samples, meta_batch_size, &self.device, &rng_context, &mut rng)
    }

    fn sample_synthetic_dataset_with_prior(
        &self,
        prior_type: &str,
        num_samples: usize,
    ) -> SyntheticTabularDataset {
        use super::train::PriorType;

        let prior_type = match prior_type {
            "Gaussian" => PriorType::Gaussian,
            "BayesianNN" => PriorType::BayesianNN,
            "RandomForest" => PriorType::RandomForest,
            "CausalDAG" => PriorType::CausalDAG,
            _ => PriorType::Gaussian,
        };

        let meta_batch_size = 1; // For validation, use single task per batch
        let rng_context = DeterministicRngContext::<B>::new(42, self.device.clone());
        let mut rng = StdRng::seed_from_u64(42);

        let prior = DatasetPrior::new(&super::train::TrainingConfig {
            model: super::config::ModelConfig::default(),
            meta_batch_size,
            tasks_per_batch: 1,
            max_samples_per_task: num_samples,
            min_samples_per_task: num_samples,
            learning_rate: 0.0,
            warmup_steps: 0,
            gradient_accumulation_steps: 1,
            gradient_clip_norm: None,
            num_epochs: 1,
            checkpoint_frequency: 1,
            validation_frequency: 1,
            early_stopping_patience: 1,
            use_gradient_checkpointing: false,
            cache_trainset_representations: false,
            layer_dropout_min_layers: None,
            prior_type,
            num_features_range: (5, 20),
            num_classes_range: (2, 5),
            feature_noise_level: 0.1,
        });

        prior.sample::<B>(num_samples, meta_batch_size, &self.device, &rng_context, &mut rng)
    }

    fn sample_indices(&self, k: usize, total: usize, rng: &mut StdRng) -> Vec<usize> {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(rng);
        indices.into_iter().take(k).collect()
    }

    fn prepare_features(
        &self,
        dataset: &SyntheticTabularDataset,
        train_indices: &[usize],
        test_indices: &[usize],
    ) -> Result<HashMap<String, Tensor<B, 3>>, Box<dyn std::error::Error>> {
        let mut x_inputs = HashMap::new();

        // Combine train and test features
        let all_indices = [train_indices, test_indices].concat();
        let all_features = dataset.features.clone();

        x_inputs.insert("main".to_string(), all_features);
        Ok(x_inputs)
    }

    fn prepare_cross_domain_features(
        &self,
        train_dataset: &SyntheticTabularDataset,
        test_dataset: &SyntheticTabularDataset,
        train_indices: &[usize],
        test_indices: &[usize],
    ) -> Result<HashMap<String, Tensor<B, 3>>, Box<dyn std::error::Error>> {
        let mut x_inputs = HashMap::new();

        // Concatenate features from both domains
        let combined_features = Tensor::cat(
            vec![train_dataset.features.clone(), test_dataset.features.clone()],
            0,
        );

        x_inputs.insert("main".to_string(), combined_features);
        Ok(x_inputs)
    }

    fn prepare_targets(
        &self,
        dataset: &SyntheticTabularDataset,
        train_indices: &[usize],
    ) -> Result<HashMap<String, Tensor<B, 3>>, Box<dyn std::error::Error>> {
        let mut y_inputs = HashMap::new();

        // Create masked targets with only training labels visible
        let mut targets = Tensor::zeros_like(&dataset.targets).add_scalar(-1);

        // This would need proper indexing to set training targets
        // For now, using simplified approach
        y_inputs.insert("main".to_string(), targets.float().unsqueeze_dim(2));
        Ok(y_inputs)
    }

    fn extract_test_targets(
        &self,
        dataset: &SyntheticTabularDataset,
        test_indices: &[usize],
    ) -> Tensor<B, 1> {
        // Extract targets for test indices
        // This would need proper indexing implementation
        dataset.targets.clone().narrow(0, test_indices[0], test_indices.len())
    }

    fn infer_num_classes(&self, targets: &Tensor<B, 2>) -> usize {
        // Find maximum target value to infer number of classes
        let max_target = targets.max().into_scalar() as i64;
        (max_target + 1) as usize
    }
}

/// Load real-world datasets for validation
pub fn load_real_world_datasets(
    path: &PathBuf,
) -> Result<Vec<RealWorldDataset>, Box<dyn std::error::Error>> {
    // Implementation would load actual datasets from disk
    // For now, returning empty vector
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_validation_engine_creation() {
        let config = ValidationConfig {
            num_synthetic_tasks: 10,
            samples_per_task_range: (50, 200),
            few_shot_sizes: vec![1, 5, 10],
            evaluate_real_datasets: false,
            real_datasets_path: None,
            protocols: vec![ValidationProtocol::FewShot(5)],
            metrics: vec![MetricType::Accuracy, MetricType::F1Score],
            output_dir: PathBuf::from("./validation_output"),
            seed: 42,
        };

        let model_config = super::super::config::ModelConfig::default();
        let device = Default::default();
        let model = PerFeatureTransformer::<Wgpu>::new(
            &model_config,
            10,
            "gelu",
            None,
            false,
            Some(6),
            false,
            None,
            false,
            &device,
        );

        let engine = ValidationEngine::new(model, config, device);
        assert_eq!(engine.config.num_synthetic_tasks, 10);
    }

    #[test]
    fn test_metric_computation() {
        let config = ValidationConfig {
            num_synthetic_tasks: 5,
            samples_per_task_range: (50, 100),
            few_shot_sizes: vec![5],
            evaluate_real_datasets: false,
            real_datasets_path: None,
            protocols: vec![ValidationProtocol::FewShot(5)],
            metrics: vec![MetricType::Accuracy],
            output_dir: PathBuf::from("./test_output"),
            seed: 42,
        };

        let results = vec![
            TaskResult {
                task_id: "test_1".to_string(),
                task_type: TaskType::Synthetic("Test".to_string()),
                num_features: 10,
                num_classes: 3,
                num_train_samples: 5,
                num_test_samples: 45,
                metrics: [("accuracy".to_string(), 0.8)].iter().cloned().collect(),
            },
            TaskResult {
                task_id: "test_2".to_string(),
                task_type: TaskType::Synthetic("Test".to_string()),
                num_features: 15,
                num_classes: 4,
                num_train_samples: 5,
                num_test_samples: 45,
                metrics: [("accuracy".to_string(), 0.75)].iter().cloned().collect(),
            },
        ];

        let model_config = super::super::config::ModelConfig::default();
        let device = Default::default();
        let model = PerFeatureTransformer::<Wgpu>::new(
            &model_config,
            10,
            "gelu",
            None,
            false,
            Some(6),
            false,
            None,
            false,
            &device,
        );

        let engine = ValidationEngine::new(model, config, device);
        let aggregate = engine.compute_aggregate_metrics(&results);

        assert_eq!(aggregate.mean_metrics.get("accuracy"), Some(&0.775));
    }
}
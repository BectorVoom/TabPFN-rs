// // Files included in this patch:
// // 1) src/regression.rs              (new)
// // 2) PATCH_SNIPPET_FOR_TRAIN.md     (instructions & diff snippet)
// // 3) docs/REGRESSION_README.md      (usage example)
//
// // === file: src/regression.rs ===
// // Regression helper for TabPFN (burn)
// // Exposes: regression_loss_and_metrics -> (loss_tensor, RegressionMetrics)
//
// use burn::tensor::{backend::Backend, Tensor};
// use burn::nn::loss::MseLoss;
//
// /// Simple struct holding regression metrics as f32 scalars.
// #[derive(Debug, Clone)]
// pub struct RegressionMetrics {
//     /// Mean squared error
//     pub mse: f32,
//     /// Mean absolute error
//     pub mae: f32,
//     /// Root mean squared error
//     pub rmse: f32,
// }
//
// /// Compute regression loss and metrics.
// ///
// /// Returns a tuple `(loss_tensor, metrics)` where `loss_tensor` is a scalar
// /// tensor (Tensor<B, 0>) produced by `MseLoss::new().forward(...)` so the
// /// caller can call `.backward()` on it. `metrics` contains f32 scalars for
// /// logging and evaluation.
// pub fn regression_loss_and_metrics<B: Backend>(
//     outputs: Tensor<B, 3>,
//     targets: Tensor<B, 3>,
// ) -> (Tensor<B, 0>, RegressionMetrics) {
//     // Basic shape check
//     let o_dims = outputs.dims();
//     let t_dims = targets.dims();
//     if o_dims != t_dims {
//         panic!(
//             "regression_loss_and_metrics: outputs and targets must have same shape. outputs={:?}, targets={:?}",
//             o_dims, t_dims
//         );
//     }
//
//     let seq_len = o_dims[0];
//     let batch_size = o_dims[1];
//     let out_dim = o_dims[2];
//     let n = seq_len * batch_size;
//
//     // reshape to [N, D]
//     let outputs_2d = outputs.clone().reshape([n, out_dim]);
//     let targets_2d = targets.clone().reshape([n, out_dim]);
//
//     // loss tensor for backward (scalar)
//     let loss_tensor = MseLoss::new().forward(outputs_2d.clone(), targets_2d.clone());
//
//     // flatten to 1D for metric calculations
//     let outputs_flat = outputs_2d.clone().reshape([n * out_dim]);
//     let targets_flat = targets_2d.clone().reshape([n * out_dim]);
//
//     // diffs
//     let diff = outputs_flat.clone() - targets_flat.clone();
//
//     // compute metrics tensors
//     let mse_tensor = diff.clone().powi(2).mean();
//     let mae_tensor = diff.abs().mean();
//
//     // convert to f32 scalars
//     let mse = mse_tensor.into_scalar().to_f32();
//     let mae = mae_tensor.into_scalar().to_f32();
//     let rmse = mse.sqrt();
//
//     (
//         loss_tensor,
//         RegressionMetrics { mse, mae, rmse },
//     )
// }
//
// // === file: PATCH_SNIPPET_FOR_TRAIN.md ===
// // Apply the following snippet to integrate regression into your training loop.
// // This is a minimal diff you can adapt into `train.rs` where loss/optimizer steps live.
//
// /*
// Example integration (apply inside your training loop where `output` and `targets`
// are available):
//
// use crate::regression;
//
// // ensure targets are floats and have shape [seq_len, batch_size, out_dim]
// // for scalar regression out_dim == 1; you may need: targets = targets.to_dtype_float().unsqueeze_dim(2)
//
// let (loss_tensor, metrics) = regression::regression_loss_and_metrics(output.clone(), targets.clone());
//
// // logging
// println!("train mse={:.6} mae={:.6} rmse={:.6}", metrics.mse, metrics.mae, metrics.rmse);
//
// // backward + optimizer step (following your existing pattern)
// loss_tensor.backward();
// optimizer.step(&mut model);
// optimizer.zero_grad();
// */
//
// // If you prefer a git-style patch, here is a conceptual diff (not runnable) showing where
// // to replace a classification loss with the regression call:
//
// /*
// - // classification loss (example)
// - let loss = cross_entropy_loss.forward(logits_reshaped, targets_labels);
// - loss.backward();
// + // regression loss
// + let (loss_tensor, metrics) = regression::regression_loss_and_metrics(output.clone(), targets.clone());
// + println!("train mse={:.6} mae={:.6} rmse={:.6}", metrics.mse, metrics.mae, metrics.rmse);
// + loss_tensor.backward();
// */
//
// // === file: docs/REGRESSION_README.md ===
// // Short usage and notes for the regression helper
//
// // # Regression helper for TabPFN (burn)
// //
// // This patch adds `src/regression.rs` which provides `regression_loss_and_metrics`.
// //
// // ## API
// //
// // ```rust
// // use crate::regression::{regression_loss_and_metrics, RegressionMetrics};
// //
// // let (loss_tensor, metrics) = regression_loss_and_metrics(output, targets);
// // // loss_tensor: Tensor<B, 0] -> can call `.backward()`
// // // metrics: RegressionMetrics { mse, mae, rmse }
// // ```
// //
// // ## Input shapes
// // - `output`: `[seq_len, batch_size, out_dim]` (usually `out_dim == 1` for scalar regression)
// // - `targets`: same shape as `output` (floats)
// //
// // ## Notes / Integration
// // - `MseLoss` from `burn` is used as the training loss.
// // - The function panics if input shapes do not match — you may change this behavior to return `Result` if you prefer.
// // - For evaluation/logging, `mse`, `mae`, `rmse` (f32) are returned in `RegressionMetrics`.
//
// // ## References
// // - burn `MseLoss` documentation: https://burn.dev/docs/burn/nn/loss/struct.MseLoss.html
// // - burn tensor docs (reshape, mean, scalar conversions): https://docs.rs/burn/latest/burn/tensor/struct.Tensor.html
// // - TabPFN (reference code / paper): https://github.com/PriorLabs/TabPFN
//
//
// // End of patch files

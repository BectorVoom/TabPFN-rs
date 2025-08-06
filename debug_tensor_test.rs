use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn main() {
    let device = Default::default();
    
    // Create a [3,2,3] tensor like in the test
    let data = TensorData::new(
        vec\![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  // First sequence
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,  // Second sequence
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0  // Third sequence
        ], 
        [3, 2, 3]
    );
    let tensor: Tensor<TestBackend, 3> = Tensor::from_data(data, &device);
    
    println\!("Original tensor shape: {:?}", tensor.dims());
    
    // Test sum_dim(0)
    let summed = tensor.clone().sum_dim(0);
    println\!("After sum_dim(0) shape: {:?}", summed.dims());
    
    // Test unsqueeze_dim(0)
    let unsqueezed = summed.unsqueeze_dim(0);
    println\!("After unsqueeze_dim(0) shape: {:?}", unsqueezed.dims());
}
EOF < /dev/null
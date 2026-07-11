use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::sigmoid,
};

/// Sinusoidal time embedding module to map scalar timesteps to high-dimensional embeddings.
#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    dim: usize,
}

impl<B: Backend> TimeEmbedding<B> {
    pub fn new(device: &B::Device, dim: usize) -> Self {
        // dim is the base embedding dimension (e.g. 32 or 64).
        // The MLP projects it to dim * 4.
        let linear_1 = LinearConfig::new(dim, dim * 4).init(device);
        let linear_2 = LinearConfig::new(dim * 4, dim * 4).init(device);
        Self { linear_1, linear_2, dim }
    }

    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        let device = timesteps.device();
        let half_dim = self.dim / 2;
        
        // Compute frequencies for sinusoidal embeddings
        let exponents = Tensor::<B, 1>::arange(0..half_dim, &device)
            .float()
            .mul_scalar(-f32::ln(10000.0) / (half_dim as f32));
        let frequencies = exponents.exp();
        
        // Reshape timesteps to [batch_size, 1] and frequencies to [1, half_dim]
        let timesteps = timesteps.unsqueeze_dim::<2>(1); // [B, 1]
        let frequencies = frequencies.unsqueeze_dim::<2>(0); // [1, half_dim]
        
        // Matrix multiply to get [batch_size, half_dim] arguments
        let arguments = timesteps.matmul(frequencies);
        
        let sin = arguments.clone().sin();
        let cos = arguments.cos();
        
        // Concatenate sin and cos along dimension 1 to get [batch_size, dim]
        let emb = Tensor::cat(vec![sin, cos], 1);
        
        // Feed forward through MLP: x * sigmoid(x) represents SiLU
        let x = self.linear_1.forward(emb);
        let x = x.clone() * sigmoid(x);
        self.linear_2.forward(x)
    }
}

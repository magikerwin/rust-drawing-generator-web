use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Linear, LinearConfig, Embedding, EmbeddingConfig,
        GroupNorm, GroupNormConfig,
    },
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

/// Class conditioning embedding module to embed class labels.
#[derive(Module, Debug)]
pub struct ClassEmbedding<B: Backend> {
    embedding: Embedding<B>,
    linear: Linear<B>,
}

impl<B: Backend> ClassEmbedding<B> {
    pub fn new(device: &B::Device, num_classes: usize, dim: usize) -> Self {
        let embedding = EmbeddingConfig::new(num_classes, dim).init(device);
        let linear = LinearConfig::new(dim, dim * 4).init(device);
        Self { embedding, linear }
    }

    pub fn forward(&self, class_ids: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let x = self.embedding.forward(class_ids);
        self.linear.forward(x)
    }
}

/// Residual Block in the U-Net that incorporates time/class conditioning.
#[derive(Module, Debug)]
pub struct UNetBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: GroupNorm<B>,
    conv2: Conv2d<B>,
    norm2: GroupNorm<B>,
    time_mlp: Linear<B>,
    shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> UNetBlock<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize, cond_dim: usize) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let num_groups = usize::min(8, out_channels);
        let norm1 = GroupNormConfig::new(num_groups, out_channels).init(device);
        
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let norm2 = GroupNormConfig::new(num_groups, out_channels).init(device);
        
        let time_mlp = LinearConfig::new(cond_dim, out_channels).init(device);
        
        let shortcut = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };
        
        Self {
            conv1,
            norm1,
            conv2,
            norm2,
            time_mlp,
            shortcut,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, cond: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x.clone());
        let h = self.norm1.forward(h);
        let mut h = h.clone() * sigmoid(h); // SiLU
        
        // Add time/class conditioning embedding
        let cond_proj = self.time_mlp.forward(cond)
            .unsqueeze_dim::<4>(2)
            .unsqueeze_dim::<4>(3); // [B, out_channels, 1, 1]
        h = h + cond_proj;
        
        let h = self.conv2.forward(h);
        let h = self.norm2.forward(h);
        let h = h.clone() * sigmoid(h); // SiLU
        
        let shortcut_out = match &self.shortcut {
            Some(conv) => conv.forward(x),
            None => x,
        };
        
        h + shortcut_out
    }
}


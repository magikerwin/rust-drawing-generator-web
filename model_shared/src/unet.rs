use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
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
        
        // Compute frequencies for sinusoidal embeddings using Int arange
        let exponents = Tensor::<B, 1, Int>::arange(0..half_dim as i64, &device)
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
    dim: usize,
}

impl<B: Backend> ClassEmbedding<B> {
    pub fn new(device: &B::Device, num_classes: usize, dim: usize) -> Self {
        let embedding = EmbeddingConfig::new(num_classes, dim).init(device);
        let linear = LinearConfig::new(dim, dim * 4).init(device);
        Self { embedding, linear, dim }
    }

    pub fn forward(&self, class_ids: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let shape = class_ids.shape();
        let batch_size = shape.dims[0];
        
        // Expose to 2D for Embedding layer
        let x = class_ids.unsqueeze_dim::<2>(1); // [B, 1]
        let x = self.embedding.forward(x); // [B, 1, dim]
        
        // Reshape back to 2D [B, dim]
        let x = x.reshape([batch_size, self.dim]);
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

/// A lightweight U-Net architecture for generating 28x28 drawings.
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    time_embed: TimeEmbedding<B>,
    class_embed: ClassEmbedding<B>,
    
    // Encoder (Downsampling)
    conv_in: Conv2d<B>,
    down_block1: UNetBlock<B>,
    downsample1: Conv2d<B>,
    down_block2: UNetBlock<B>,
    downsample2: Conv2d<B>,
    
    // Bottleneck
    bottleneck_block: UNetBlock<B>,
    
    // Decoder (Upsampling)
    up1: ConvTranspose2d<B>,
    up_block1: UNetBlock<B>,
    up2: ConvTranspose2d<B>,
    up_block2: UNetBlock<B>,
    
    // Output
    conv_out: Conv2d<B>,
}

impl<B: Backend> UNet<B> {
    pub fn new(device: &B::Device, num_classes: usize, base_dim: usize) -> Self {
        let cond_dim = base_dim * 4;
        
        let time_embed = TimeEmbedding::new(device, base_dim);
        let class_embed = ClassEmbedding::new(device, num_classes, base_dim);
        
        let conv_in = Conv2dConfig::new([1, base_dim], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let down_block1 = UNetBlock::new(device, base_dim, base_dim, cond_dim);
        
        let downsample1 = Conv2dConfig::new([base_dim, base_dim * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let down_block2 = UNetBlock::new(device, base_dim * 2, base_dim * 2, cond_dim);
        
        let downsample2 = Conv2dConfig::new([base_dim * 2, base_dim * 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        let bottleneck_block = UNetBlock::new(device, base_dim * 4, base_dim * 4, cond_dim);
        
        let up1 = ConvTranspose2dConfig::new([base_dim * 4, base_dim * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);
        
        let up_block1 = UNetBlock::new(device, base_dim * 4, base_dim * 2, cond_dim);
        
        let up2 = ConvTranspose2dConfig::new([base_dim * 2, base_dim], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);
        
        let up_block2 = UNetBlock::new(device, base_dim * 2, base_dim, cond_dim);
        
        let conv_out = Conv2dConfig::new([base_dim, 1], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        
        Self {
            time_embed,
            class_embed,
            conv_in,
            down_block1,
            downsample1,
            down_block2,
            downsample2,
            bottleneck_block,
            up1,
            up_block1,
            up2,
            up_block2,
            conv_out,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, timesteps: Tensor<B, 1>, class_ids: Tensor<B, 1, Int>) -> Tensor<B, 4> {
        let time_emb = self.time_embed.forward(timesteps);
        let class_emb = self.class_embed.forward(class_ids);
        let cond = time_emb + class_emb;
        
        // Encoder
        let x1 = self.conv_in.forward(x);
        let x1 = self.down_block1.forward(x1, cond.clone());
        
        let x2 = self.downsample1.forward(x1.clone());
        let x2 = self.down_block2.forward(x2, cond.clone());
        
        let x3 = self.downsample2.forward(x2.clone());
        let x3 = self.bottleneck_block.forward(x3, cond.clone());
        
        // Decoder
        let u1 = self.up1.forward(x3);
        let u1 = Tensor::cat(vec![u1, x2], 1);
        let u1 = self.up_block1.forward(u1, cond.clone());
        
        let u2 = self.up2.forward(u1);
        let u2 = Tensor::cat(vec![u2, x1], 1);
        let u2 = self.up_block2.forward(u2, cond);
        
        self.conv_out.forward(u2)
    }
}

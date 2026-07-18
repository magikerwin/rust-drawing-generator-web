use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Linear, LinearConfig, Embedding, EmbeddingConfig,
        GroupNorm, GroupNormConfig,
    },
    prelude::*,
    tensor::activation::{sigmoid, softmax},
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
        // Allocate num_classes + 1 slots to store the unconditional class embedding at index `num_classes`
        let embedding = EmbeddingConfig::new(num_classes + 1, dim).init(device);
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
        let batch_size = x.shape().dims[0];
        let out_channels = h.shape().dims[1];
        let cond_proj = self.time_mlp.forward(cond)
            .reshape([batch_size, out_channels, 1, 1]);
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

/// Single-head Self-Attention block for low-resolution feature maps (e.g. 7x7 bottleneck).
/// Learns global spatial relationships with O(N^2) complexity where N = H * W.
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    channels: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(device: &B::Device, channels: usize) -> Self {
        let q_proj = LinearConfig::new(channels, channels).init(device);
        let k_proj = LinearConfig::new(channels, channels).init(device);
        let v_proj = LinearConfig::new(channels, channels).init(device);
        let out_proj = LinearConfig::new(channels, channels).init(device);
        Self { q_proj, k_proj, v_proj, out_proj, channels }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let channels = shape.dims[1];
        let h = shape.dims[2];
        let w = shape.dims[3];
        let n = h * w;

        // Flatten spatial dimensions [B, C, H, W] -> [B, C, N] and transpose to [B, N, C]
        let x_flat = x.clone().reshape([batch_size, channels, n]).swap_dims(1, 2);

        // Project Queries, Keys, and Values: shape [B, N, C]
        let q = self.q_proj.forward(x_flat.clone());
        let k = self.k_proj.forward(x_flat.clone());
        let v = self.v_proj.forward(x_flat);

        // Compute scaled dot-product attention scores: [B, N, N]
        let k_t = k.swap_dims(1, 2);
        let scale = (channels as f64).sqrt();
        let scores = q.matmul(k_t).div_scalar(scale);

        // Softmax normalization across rows: [B, N, N]
        let attn_map = softmax(scores, 2);

        // Average values weighted by attention map: [B, N, C]
        let out = attn_map.matmul(v);

        // Project output and swap back to original shape [B, C, H, W]
        let out = self.out_proj.forward(out);
        let out = out.swap_dims(1, 2).reshape([batch_size, channels, h, w]);

        // Residual skip connection
        x + out
    }
}

/// A lightweight U-Net architecture for generating 28x28 drawings.
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    pub num_classes: usize,
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
    bottleneck_attn: SelfAttention<B>,
    
    // Decoder (Upsampling)
    up1: ConvTranspose2d<B>,
    up_block1: UNetBlock<B>,
    decoder_attn: SelfAttention<B>,
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
        let bottleneck_attn = SelfAttention::new(device, base_dim * 4);
        
        let up1 = ConvTranspose2dConfig::new([base_dim * 4, base_dim * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .with_padding_out([1, 1])
            .init(device);
        
        let up_block1 = UNetBlock::new(device, base_dim * 4, base_dim * 2, cond_dim);
        let decoder_attn = SelfAttention::new(device, base_dim * 2);
        
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
            num_classes,
            time_embed,
            class_embed,
            conv_in,
            down_block1,
            downsample1,
            down_block2,
            downsample2,
            bottleneck_block,
            bottleneck_attn,
            up1,
            up_block1,
            decoder_attn,
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
        let x3 = self.bottleneck_attn.forward(x3);
        
        // Decoder
        let u1 = self.up1.forward(x3);
        let u1 = Tensor::cat(vec![u1, x2], 1);
        let u1 = self.up_block1.forward(u1, cond.clone());
        let u1 = self.decoder_attn.forward(u1);
        
        let u2 = self.up2.forward(u1);
        let u2 = Tensor::cat(vec![u2, x1], 1);
        let u2 = self.up_block2.forward(u2, cond);
        
        self.conv_out.forward(u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_self_attention_shapes() {
        let device = Default::default();
        let attn = SelfAttention::<NdArray>::new(&device, 32);
        let input = Tensor::<NdArray, 4>::random([2, 32, 7, 7], burn::tensor::Distribution::Default, &device);
        let output = attn.forward(input.clone());
        assert_eq!(output.shape().dims, input.shape().dims);
    }
}

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        LayerNorm, LayerNormConfig,
        PaddingConfig2d,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::relu,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // Block 1: 28x28 -> 14x14, 1 -> 16 channels
    conv1: Conv2d<B>,
    ln1: LayerNorm<B>,
    proj1: Conv2d<B>,

    // Block 2: 14x14 -> 7x7, 16 -> 32 channels
    conv2: Conv2d<B>,
    ln2: LayerNorm<B>,
    proj2: Conv2d<B>,

    // Block 3: Stays at 7x7, 32 -> 64 channels
    conv3: Conv2d<B>,
    ln3: LayerNorm<B>,
    proj3: Conv2d<B>,

    // Classifier: GAP output (64 channels) -> logits
    fc: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // Block 1
        let conv1 = Conv2dConfig::new([1, 16], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let ln1 = LayerNormConfig::new(16).init(device);
        let proj1 = Conv2dConfig::new([1, 16], [1, 1])
            .with_stride([2, 2])
            .init(device);

        // Block 2
        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let ln2 = LayerNormConfig::new(32).init(device);
        let proj2 = Conv2dConfig::new([16, 32], [1, 1])
            .with_stride([2, 2])
            .init(device);

        // Block 3 (No downsampling, stride 1)
        let conv3 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let ln3 = LayerNormConfig::new(64).init(device);
        let proj3 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .init(device);

        // Classifier: Single Linear layer mapping GAP features to class logits
        let fc = LinearConfig::new(64, num_classes).init(device);

        Self {
            conv1,
            ln1,
            proj1,
            conv2,
            ln2,
            proj2,
            conv3,
            ln3,
            proj3,
            fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Block 1: Conv (stride 2) + Proj (stride 2)
        let y = self.conv1.forward(input.clone());
        let y = y.swap_dims(1, 3);
        let y = self.ln1.forward(y);
        let y = y.swap_dims(1, 3);
        
        let shortcut = self.proj1.forward(input);
        let x = relu(y + shortcut);

        // Block 2: Conv (stride 2) + Proj (stride 2)
        let y = self.conv2.forward(x.clone());
        let y = y.swap_dims(1, 3);
        let y = self.ln2.forward(y);
        let y = y.swap_dims(1, 3);

        let shortcut = self.proj2.forward(x);
        let x = relu(y + shortcut);

        // Block 3: Conv (stride 1) + Proj (stride 1)
        let y = self.conv3.forward(x.clone());
        let y = y.swap_dims(1, 3);
        let y = self.ln3.forward(y);
        let y = y.swap_dims(1, 3);

        let shortcut = self.proj3.forward(x);
        let x = relu(y + shortcut);

        // Global Average Pooling: [Batch, 64, 7, 7] -> [Batch, 64, 1, 1]
        let x = x.mean_dim(2).mean_dim(3);

        // Reshape: [Batch, 64, 1, 1] -> [Batch, 64]
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 64]);

        // Classifier projection directly to logits
        self.fc.forward(x)
    }
}

use burn::module::Module;
use burn::prelude::*;
use model_shared::UNet;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub unet: UNet<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // We set base_dim to 32 channels. This provides a great balance of
        // generation quality and lightweight CPU inference speed.
        Self {
            unet: UNet::new(device, num_classes, 32),
        }
    }
}

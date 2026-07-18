pub mod unet;
pub mod scheduler;

pub use unet::UNet;
pub use scheduler::DDIMScheduler;

use burn::module::Module;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub unet: UNet<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        Self {
            unet: UNet::new(device, num_classes, 48),
        }
    }
}


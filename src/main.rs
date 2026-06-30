mod model;
mod data;
mod training;

use burn::{
    backend::{Autodiff, NdArray},
    data::dataset::vision::MnistDataset,
    optim::AdamConfig,
};
use crate::training::{train, TrainingConfig};

fn main() {
    // 1. Define where training checkpoints and config will be saved
    let artifact_dir = "./target/mnist-model";

    // 2. Select CPU backend with Autodiff enabled for training
    type MyBackend = Autodiff<NdArray>;
    let device = Default::default(); // default to CPU device

    println!("Starting MNIST training on CPU (NdArray backend)...");

    // 3. Kick off training using the train orchestrator
    train::<MyBackend, _, _>(
        artifact_dir,
        TrainingConfig::new(AdamConfig::new()),
        device,
        MnistDataset::train(),
        MnistDataset::test(),
    );

    println!("Training finished! Model saved successfully to {}", artifact_dir);
}

mod model;
mod data;
mod training;
mod inference;

use burn::{
    backend::{Autodiff, NdArray},
    data::dataset::{vision::MnistDataset, Dataset},
    optim::AdamConfig,
};
use crate::training::{train, TrainingConfig};
use crate::inference::{load_model, predict};

fn main() {
    let artifact_dir = "./target/mnist-model";

    // Simple argument parsing to toggle between training and inference
    let args: Vec<String> = std::env::args().collect();
    let run_inference = args.contains(&"--predict".to_string());

    if run_inference {
        println!("Loading model for inference...");
        let device = Default::default(); // NdArray CPU device
        let model = load_model(artifact_dir, &device);

        // Fetch a sample from the MNIST test dataset
        let test_dataset = MnistDataset::test();
        let sample_index = 0; // Pick the first test image
        let sample = test_dataset.get(sample_index).expect("Failed to get sample");

        // Flatten the 28x28 image grid into a 784 array
        let mut flattened_image = [0.0f32; 784];
        for i in 0..28 {
            for j in 0..28 {
                flattened_image[i * 28 + j] = sample.image[i][j];
            }
        }

        // Draw a simple ASCII art representing the input digit
        println!("\nInput Image:");
        for i in 0..28 {
            for j in 0..28 {
                if sample.image[i][j] > 0.5 {
                    print!("#");
                } else if sample.image[i][j] > 0.1 {
                    print!(".");
                } else {
                    print!(" ");
                }
            }
            println!();
        }

        // Perform prediction
        let predicted_digit = predict(&model, flattened_image, &device);
        println!("\nTarget Label (Ground Truth): {}", sample.label);
        println!("Model Prediction           : {}", predicted_digit);

    } else {
        type MyBackend = Autodiff<NdArray>;
        let device = Default::default();

        println!("Starting MNIST training on CPU (NdArray backend)...");
        train::<MyBackend, _, _>(
            artifact_dir,
            TrainingConfig::new(AdamConfig::new()),
            device,
            MnistDataset::train(),
            MnistDataset::test(),
        );
        println!("Training finished! Model saved successfully to {}", artifact_dir);
    }
}

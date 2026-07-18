use crate::{
    data::{MnistBatch, MnistBatcher},
    model::Model,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::Dataset,
    },
    nn::loss::{MseLoss, Reduction},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, BinFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric,
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

/// Configuration struct holding the training hyperparameters.
#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,      // Number of epochs to train for
    #[config(default = 32)]
    pub batch_size: usize,      // Number of samples in each batch
    #[config(default = 1)]
    pub num_workers: usize,     // Number of parallel threads used for data loading
    #[config(default = 42)]
    pub seed: u64,              // Seed for random number generators (reproducibility)
    pub optimizer: AdamConfig,  // Optimizer configuration (e.g. learning rate, betas)
    #[config(default = 2e-4)]
    pub learning_rate: f64,     // Static learning rate for training
    #[config(default = "String::from(\"noise\")")]
    pub prediction_type: String, // Target type ("noise" or "velocity")
}

/// Orchestrates the training pipeline. Decoupled from specific datasets using generic types.
pub fn train<B: AutodiffBackend, D1, D2>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    train_dataset: D1,
    valid_dataset: D2,
    num_classes: usize,
    allow_horizontal_flip: bool,
) where
    D1: Dataset<burn::data::dataset::vision::MnistItem> + 'static,
    D2: Dataset<burn::data::dataset::vision::MnistItem> + 'static,
{
    let config_path = format!("{artifact_dir}/config.json");
    if std::path::Path::new(&config_path).exists() {
        if let Ok(existing_content) = std::fs::read_to_string(&config_path) {
            if let Ok(new_content) = serde_json::to_string(&config) {
                if existing_content != new_content {
                    println!("\n==========================================================================");
                    println!("WARNING: Training configuration or model architecture mismatch detected!");
                    println!("Existing checkpoints in '{}' may be incompatible.", artifact_dir);
                    println!("Please delete this directory to start a clean training session:");
                    println!("  Remove-Item -Recurse -Force {} (Windows PowerShell)", artifact_dir);
                    println!("  rm -rf {} (Linux / macOS)", artifact_dir);
                    println!("==========================================================================\n");
                    panic!("Configuration mismatch. Clear the artifact directory to continue.");
                }
            }
        }
    }

    std::fs::create_dir_all(artifact_dir).ok();
    config.save(&config_path).expect("Save config failed");

    // Set the backend random seed for reproducible initialization and shuffling
    B::seed(config.seed);

    // Initialize the batcher for training data, passing target type and num_classes
    let batcher_train = MnistBatcher::<B>::new(device.clone(), true, allow_horizontal_flip, config.prediction_type.clone(), num_classes);
    
    // Initialize the batcher for validation data, passing target type and num_classes
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone(), false, false, config.prediction_type.clone(), num_classes);

    // Build the training DataLoader
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    // Build the validation DataLoader
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Configure and build the training driver (Learner)
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())      // Track training loss
        .metric_valid_numeric(LossMetric::new())      // Track validation loss
        .with_file_checkpointer(CompactRecorder::new()) // Save training checkpointers on disk
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            Model::<B>::new(&device, num_classes), // Instantiate the Model wrapping UNet
            config.optimizer.init(),  // Initialize the optimizer state
            config.learning_rate,    // Learning rate for training the diffusion model
        );

    // Start the training and validation loops
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save the final trained model parameters (weights and biases) to disk
    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model saving failed");

    model_trained
        .save_file(format!("{artifact_dir}/model"), &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Model saving with BinFileRecorder failed");
}

/// Implement the TrainStep trait for the Model to specify how it performs a single training iteration.
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.unet.forward(
            batch.corrupted_images,
            batch.timesteps,
            batch.targets,
        );
        let loss = MseLoss::new().forward(
            output.clone(),
            batch.target.clone(),
            Reduction::Auto,
        );

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput::new(
                loss,
                output.flatten(1, 3),
                batch.target.flatten(1, 3),
            ),
        )
    }
}

/// Implement the ValidStep trait for the Model to specify how it evaluates validation data.
impl<B: Backend> ValidStep<MnistBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> RegressionOutput<B> {
        let output = self.unet.forward(
            batch.corrupted_images,
            batch.timesteps,
            batch.targets,
        );
        let loss = MseLoss::new().forward(
            output.clone(),
            batch.target.clone(),
            Reduction::Auto,
        );

        RegressionOutput::new(
            loss,
            output.flatten(1, 3),
            batch.target.flatten(1, 3),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::data::dataset::{vision::MnistItem, InMemDataset};

    #[test]
    fn test_train() {
        let artifact_dir = "./target/test-mnist-model";
        std::fs::remove_dir_all(artifact_dir).ok(); // Clean up any leftovers from previous aborted runs
        type TestBackend = Autodiff<NdArray>;
        let device = Default::default();

        // 1. Create a tiny mock dataset in memory (2 items)
        let item1 = MnistItem {
            image: [[0.0; 28]; 28],
            label: 3,
        };
        let item2 = MnistItem {
            image: [[1.0; 28]; 28],
            label: 7,
        };
        let train_dataset = InMemDataset::new(vec![item1.clone(), item2.clone()]);
        let valid_dataset = InMemDataset::new(vec![item1, item2]);

        // 2. Configure training for 1 epoch, batch size 2, 1 worker thread
        let mut config = TrainingConfig::new(AdamConfig::new());
        config.num_epochs = 1;
        config.batch_size = 2;
        config.num_workers = 1;

        // 3. Dry-run training on the mock dataset (will finish instantly)
        train::<TestBackend, _, _>(artifact_dir, config, device, train_dataset, valid_dataset, 10, false);

        // 4. Verify that parameters were successfully optimized and saved
        assert!(std::path::Path::new(&format!("{artifact_dir}/model.mpk")).exists() 
            || std::path::Path::new(&format!("{artifact_dir}/model.bin")).exists()
            || std::path::Path::new(&format!("{artifact_dir}/model")).exists());

        // 5. Clean up test outputs
        std::fs::remove_dir_all(artifact_dir).ok();
    }
}

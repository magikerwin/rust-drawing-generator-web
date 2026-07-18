use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};
use rand::Rng;
use model_shared::DDIMScheduler;

// Clone is required by the DataLoader to send copies of the batcher to multiple worker threads.
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
    is_training: bool,
    allow_flip: bool,
    scheduler: std::sync::Arc<DDIMScheduler>,
    prediction_type: String, // "noise" (DDPM) or "velocity" (Flow Matching)
    num_classes: usize,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device, is_training: bool, allow_flip: bool, prediction_type: String, num_classes: usize) -> Self {
        // Standard 1000-step linear schedule for training
        let scheduler = std::sync::Arc::new(DDIMScheduler::new(1000, 1e-4, 0.02));
        Self {
            device,
            is_training,
            allow_flip,
            scheduler,
            prediction_type,
            num_classes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub corrupted_images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
    pub timesteps: Tensor<B, 1>,
    pub target: Tensor<B, 4>, // Holds the learning target: noise (DDPM) or velocity (Flow Matching)
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let mut rng = rand::thread_rng();
        let batch_size = items.len();

        // 1. Process clean images and normalize to [-1.0, 1.0]
        let clean_images = items
            .iter()
            .map(|item| {
                let img = if self.is_training {
                    // Random shift between -2 and +2 pixels
                    let dx = rng.gen_range(-2..=2);
                    let dy = rng.gen_range(-2..=2);
                    // Random scale between 0.9 and 1.1
                    let scale = rng.gen_range(0.9..=1.1);
                    // 50% probability of horizontal flip (if allowed)
                    let flip_h = self.allow_flip && rng.gen_bool(0.5);

                    augment_image(&item.image, dx, dy, scale, flip_h)
                } else {
                    item.image
                };

                // Determine max pixel to check range ([0..1] vs [0..255])
                let max_pixel = img.iter().flat_map(|row| row.iter()).fold(0.0f32, |m, &x| m.max(x));
                let mut img_normalized = [[0.0f32; 28]; 28];
                for y in 0..28 {
                    for x in 0..28 {
                        let val = img[y][x];
                        img_normalized[y][x] = if max_pixel > 1.0 {
                            (val / 127.5) - 1.0
                        } else {
                            (val * 2.0) - 1.0
                        };
                    }
                }
                img_normalized
            })
            .map(|img| TensorData::from(img))
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 1, 28, 28]))
            .collect::<Vec<_>>();

        let clean_images = Tensor::cat(clean_images, 0); // [B, 1, 28, 28]

        // 2. Generate target class labels
        let targets = items
            .iter()
            .map(|item| {
                let mut label = item.label as usize;
                // Classifier-free guidance training: randomly drop the class conditioning (15% drop rate)
                if self.is_training && rng.gen_bool(0.15) {
                    label = self.num_classes;
                }
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([label as i32]),
                    &self.device,
                )
            })
            .collect::<Vec<_>>();
        let targets = Tensor::cat(targets, 0); // [B]

        // 3. Generate random timesteps t ~ U(0, 1000)
        let mut t_vec = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            t_vec.push(rng.gen_range(0..1000) as i64);
        }
        let timesteps_int = Tensor::<B, 1, Int>::from_ints(t_vec.as_slice(), &self.device);
        let timesteps = timesteps_int.clone().float(); // [B]

        // 4. Generate random Gaussian noise epsilon ~ N(0, I)
        let noise = Tensor::<B, 4>::random(
            [batch_size, 1, 28, 28],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // 5. Corrupt clean images and select the model's prediction target
        let (corrupted_images, target) = if self.prediction_type == "velocity" {
            // --- EDUCATIONAL: FLOW MATCHING TRAJECTORY ---
            // In Flow Matching, the noisy image x_t is a direct linear interpolation
            // between the clean image (x0) and the random noise (x1):
            //     x_t = (1 - t) * x_0 + t * noise
            // The model is trained to predict the straight line velocity field (v_t = noise - x0).
            let corrupted = self.scheduler.add_noise_flow(clean_images.clone(), noise.clone(), timesteps.clone());
            let target_velocity = noise - clean_images;
            (corrupted, target_velocity)
        } else {
            // --- EDUCATIONAL: DDPM TRAJECTORY ---
            // In standard DDPM, the noisy image x_t is corrupted along a curved schedule:
            //     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
            // The model is trained to predict the added noise directly.
            let corrupted = self.scheduler.add_noise(clean_images, noise.clone(), timesteps_int);
            (corrupted, noise)
        };

        MnistBatch {
            corrupted_images,
            targets,
            timesteps,
            target,
        }
    }
}

/// Nearest-neighbor image coordinate transformation for 2d scaling, translation, and flipping.
fn augment_image(
    image: &[[f32; 28]; 28],
    dx: i32,
    dy: i32,
    scale: f32,
    flip_h: bool,
) -> [[f32; 28]; 28] {
    let mut augmented = [[0.0; 28]; 28];
    for y in 0..28 {
        let src_y = ((y as f32 - 13.5) / scale + 13.5) - dy as f32;
        let src_y_i = src_y.round() as i32;

        if src_y_i >= 0 && src_y_i < 28 {
            let sy_idx = src_y_i as usize;
            for x in 0..28 {
                let src_x = ((x as f32 - 13.5) / scale + 13.5) - dx as f32;
                let src_x_i = src_x.round() as i32;

                if src_x_i >= 0 && src_x_i < 28 {
                    let mut sx = src_x_i as usize;
                    if flip_h {
                        sx = 27 - sx;
                    }
                    augmented[y][x] = image[sy_idx][sx];
                }
            }
        }
    }
    augmented
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_batcher() {
        type TestBackend = NdArray;
        let device = Default::default();
        let batcher = MnistBatcher::<TestBackend>::new(device, false, false, "noise".to_string(), 10);

        let item1 = MnistItem {
            image: [[0.0; 28]; 28],
            label: 3,
        };
        let item2 = MnistItem {
            image: [[1.0; 28]; 28],
            label: 7,
        };

        let batch = batcher.batch(vec![item1, item2]);

        let image_shape = batch.corrupted_images.shape();
        assert_eq!(image_shape.dims, [2, 1, 28, 28]);

        let target_shape = batch.targets.shape();
        assert_eq!(target_shape.dims, [2]);

        let target_tensor_shape = batch.target.shape();
        assert_eq!(target_tensor_shape.dims, [2, 1, 28, 28]);

        let timesteps_shape = batch.timesteps.shape();
        assert_eq!(timesteps_shape.dims, [2]);
    }
}

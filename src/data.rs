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
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device, is_training: bool, allow_flip: bool) -> Self {
        // Standard 1000-step linear schedule for training
        let scheduler = std::sync::Arc::new(DDIMScheduler::new(1000, 1e-4, 0.02));
        Self {
            device,
            is_training,
            allow_flip,
            scheduler,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub corrupted_images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
    pub timesteps: Tensor<B, 1>,
    pub noise: Tensor<B, 4>,
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
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([item.label as i32]),
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

        // 5. Corrupt clean images to get x_t
        let corrupted_images = self.scheduler.add_noise(clean_images, noise.clone(), timesteps_int);

        MnistBatch {
            corrupted_images,
            targets,
            timesteps,
            noise,
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
        for x in 0..28 {
            // Subtracting 13.5 shifts coordinates so scaling is centered around the middle of the 28x28 grid
            let src_x = ((x as f32 - 13.5) / scale + 13.5) - dx as f32;
            let src_y = ((y as f32 - 13.5) / scale + 13.5) - dy as f32;

            let src_x_i = src_x.round() as i32;
            let src_y_i = src_y.round() as i32;

            if src_x_i >= 0 && src_x_i < 28 && src_y_i >= 0 && src_y_i < 28 {
                let mut sx = src_x_i as usize;
                if flip_h {
                    sx = 27 - sx;
                }
                augmented[y][x] = image[src_y_i as usize][sx];
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
        let batcher = MnistBatcher::<TestBackend>::new(device, false, false);

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

        let noise_shape = batch.noise.shape();
        assert_eq!(noise_shape.dims, [2, 1, 28, 28]);

        let timesteps_shape = batch.timesteps.shape();
        assert_eq!(timesteps_shape.dims, [2]);
    }
}

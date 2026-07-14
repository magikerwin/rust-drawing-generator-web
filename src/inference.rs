use crate::model::Model;
use burn::{
    backend::NdArray,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

/// Loads the trained model weights from the artifact directory and returns the Model.
pub fn load_model(artifact_dir: &str, device: &<NdArray as Backend>::Device, num_classes: usize) -> Model<NdArray> {
    let recorder = CompactRecorder::new();
    
    // Load the saved weights using the compact recorder
    let record = recorder
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Failed to load model parameters");

    // Reconstruct the model architecture and load the weights
    Model::<NdArray>::new(device, num_classes).load_record(record)
}

/// Helper function to convert a 4D image tensor in [-1, 1] range to a 1D pixel array in [0, 255] range.
fn tensor_to_pixels<B: Backend>(tensor: Tensor<B, 4>) -> Vec<f32> {
    let data = tensor.into_data().into_vec::<f32>().expect("Failed to extract tensor data");
    data.into_iter()
        .map(|val| {
            let denorm = (val + 1.0) * 127.5;
            denorm.clamp(0.0, 255.0)
        })
        .collect()
}

/// Generates a drawing using the iterative DDIM reverse process.
/// Returns a history of intermediate images (each flattened to 784 pixels in [0, 255] range).
pub fn generate_image_steps(
    model: &Model<NdArray>,
    device: &<NdArray as Backend>::Device,
    class_id: usize,
    num_steps: usize,
    schedule: &str,
) -> Vec<Vec<f32>> {
    let scheduler = model_shared::DDIMScheduler::new(1000, 1e-4, 0.02);
    
    // Start with random Gaussian noise x_T ~ N(0, I)
    let mut x_t = Tensor::<NdArray, 4>::random(
        [1, 1, 28, 28],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    
    let class_ids = Tensor::<NdArray, 1, Int>::from_ints([class_id as i32], device);
    
    // Generate skipped timesteps for DDIM based on schedule type
    let mut steps = Vec::new();
    if schedule == "linear" {
        // Linear spacing: spreads steps evenly across the 0..1000 range.
        // Good for generic sampling, but can suffer from lack of detail refinement
        // when generating with very few total steps.
        let step_ratio = 1000 / num_steps;
        for i in (0..num_steps).rev() {
            steps.push(i * step_ratio);
        }
    } else {
        // Parse power exponent (defaults to 2.0 for quadratic)
        let rho: f32 = match schedule {
            "quadratic" => 2.0,
            other => other.parse::<f32>().unwrap_or(2.0),
        };
        // Polynomial/Power spacing: concentrates steps near t=0 using exponent rho.
        // Concentrating more steps at lower timesteps is ideal for diffusion models because
        // the final denoising steps are critical for resolving high-frequency details.
        if num_steps > 1 {
            for i in (0..num_steps).rev() {
                let x = i as f32 / (num_steps - 1) as f32;
                let t = (x.powf(rho) * 999.0).round() as usize;
                steps.push(t);
            }
        } else {
            steps.push(0);
        }
    }
    
    let mut history = Vec::with_capacity(steps.len() + 1);
    
    // Push the initial noise state
    history.push(tensor_to_pixels(x_t.clone()));
    
    // DDIM Denoising Loop
    for i in 0..steps.len() {
        let t = steps[i];
        let prev_t = if i + 1 < steps.len() {
            Some(steps[i + 1])
        } else {
            None
        };
        
        let timesteps = Tensor::<NdArray, 1>::from_floats([t as f32], device);
        
        // Predict noise using the U-Net model
        let predicted_noise = model.unet.forward(x_t.clone(), timesteps, class_ids.clone());
        
        // Denoise one step
        x_t = scheduler.step(x_t, predicted_noise, t, prev_t);
        
        // Push intermediate drawing state
        history.push(tensor_to_pixels(x_t.clone()));
    }
    
    history
}

/// Renders a 784-pixel drawing to the console as ASCII art.
pub fn render_ascii(pixels: &[f32]) {
    assert_eq!(pixels.len(), 784);
    for y in 0..28 {
        let mut line = String::new();
        for x in 0..28 {
            let val = pixels[y * 28 + x];
            let char = if val > 200.0 {
                "#"
            } else if val > 150.0 {
                "%"
            } else if val > 100.0 {
                "*"
            } else if val > 50.0 {
                "."
            } else {
                " "
            };
            line.push_str(char);
        }
        println!("{}", line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_generate_image_steps() {
        let device = Default::default();
        let model = Model::<NdArray>::new(&device, 10);
        
        // Generate with 5 steps for test performance
        let history = generate_image_steps(&model, &device, 3, 5, "linear");
        assert_eq!(history.len(), 6); // 1 initial noise state + 5 denoising steps
        assert_eq!(history[0].len(), 784);
    }
}

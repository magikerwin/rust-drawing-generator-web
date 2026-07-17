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

/// Generates a drawing using the iterative DDIM or Flow Matching reverse process.
/// Returns a history of intermediate images (each flattened to 784 pixels in [0, 255] range).
pub fn generate_image_steps(
    model: &Model<NdArray>,
    device: &<NdArray as Backend>::Device,
    class_id: usize,
    num_steps: usize,
    schedule: &str,
    sampler: &str,
    prediction_type: &str,
) -> Vec<Vec<f32>> {
    let scheduler = model_shared::DDIMScheduler::new(1000, 1e-4, 0.02);
    
    // Start with random Gaussian noise x_T ~ N(0, I)
    let mut x_t = Tensor::<NdArray, 4>::random(
        [1, 1, 28, 28],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );
    
    let class_ids = Tensor::<NdArray, 1, Int>::from_ints([class_id as i32], device);
    
    // Generate skipped timesteps based on schedule type
    let mut steps = Vec::new();
    if schedule == "linear" {
        // Linear spacing: spreads steps evenly across the 0..1000 range.
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
    
    // Sampler Denoising Loop
    for i in 0..steps.len() {
        let t = steps[i];
        let prev_t = if i + 1 < steps.len() {
            Some(steps[i + 1])
        } else {
            None
        };
        
        let timesteps = Tensor::<NdArray, 1>::from_floats([t as f32], device);
        let out_1 = model.unet.forward(x_t.clone(), timesteps, class_ids.clone());
        
        if prediction_type == "velocity" {
            // --- EDUCATIONAL: FLOW MATCHING REVERSE ODE SOLVERS ---
            // In Flow Matching, we model a velocity field v_\theta(x_t, t) = x_1 - x_0.
            // The reverse process integrates this velocity field backwards in time from
            // t = 1.0 (Gaussian noise) to t = 0.0 (the clean generated image).
            //
            // Since we count from t=1000 to t=0 in integer steps, we scale them to [0..1]
            // where t_scaled = t / 1000.0. The step size dt = t_scaled - prev_t_scaled is
            // positive, and we subtract the velocity update: x_{t-dt} = x_t - dt * v.
            
            let t_scaled = t as f32 / 1000.0;
            let prev_t_val = prev_t.unwrap_or(0);
            let prev_t_scaled = prev_t_val as f32 / 1000.0;
            let dt = t_scaled - prev_t_scaled;
            
            if sampler == "heun" && prev_t.is_some() {
                // Heun's 2nd-Order ODE Solver (Predictor-Corrector Method)
                // 1. Predictor: Estimate intermediate state x_prev_pred via standard Euler step
                let x_prev_pred = x_t.clone() - out_1.clone().mul_scalar(dt);
                
                // 2. Evaluate model's velocity field at the estimated future state
                let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], device);
                let out_2 = model.unet.forward(x_prev_pred, prev_timesteps, class_ids.clone());
                
                // 3. Corrector: Update using the average of both velocities
                let v_avg = (out_1 + out_2).mul_scalar(0.5);
                x_t = x_t.clone() - v_avg.mul_scalar(dt);
            } else {
                // Euler 1st-Order ODE Solver
                x_t = x_t.clone() - out_1.mul_scalar(dt);
            }
        } else {
            // --- DDPM / DDIM REVERSE PROCESS ---
            let epsilon_1 = out_1;
            
            if sampler == "heun" && prev_t.is_some() {
                let prev_t_val = prev_t.unwrap();
                
                // --- EDUCATIONAL: HEUN'S 2ND-ORDER SAMPLER (PREDICTOR-CORRECTOR METHOD) ---
                // While DDIM works as a 1st-order solver (Euler method) that takes steps along the 
                // initial derivative (epsilon_1), Heun's method computes a 2nd-order correction step 
                // to drastically decrease numerical approximation errors along the ODE trajectory.
                // 
                // Note: Since noise scaling factors (alphas/betas) change non-linearly at each step,
                // we cannot simply average the predicted noise vectors directly. We must compute the
                // predicted clean states (x0) at each timestep using their respective scaling factors,
                // average the states and noise, and then project the final integration.
                
                let alpha_t = scheduler.alphas_cumprod[t];
                let alpha_prev = scheduler.alphas_cumprod[prev_t_val];
                let beta_t = 1.0 - alpha_t;
                let beta_prev = 1.0 - alpha_prev;
                
                // 1. PREDICT CLEAN x0_1:
                // Predict the clean image (x0_1) from current state x_t and current noise (epsilon_1)
                let x0_1 = (x_t.clone() - epsilon_1.clone().mul_scalar(beta_t.sqrt()))
                    .div_scalar(alpha_t.sqrt());
                    
                // 2. PREDICTOR STEP (Euler Step to next state):
                // Project the state forward to an intermediate state (x_prev_pred) at the next timestep (prev_t)
                let x_prev_pred = x0_1.clone().mul_scalar(alpha_prev.sqrt()) + epsilon_1.clone().mul_scalar(beta_prev.sqrt());
                
                // 3. CORRECTOR STEP:
                // Evaluate the model's U-Net again at the estimated intermediate state (x_prev_pred)
                // at the future timestep (prev_t) to get the predicted future noise (epsilon_2).
                let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], device);
                let epsilon_2 = model.unet.forward(x_prev_pred.clone(), prev_timesteps, class_ids.clone());
                
                // 4. PREDICT CLEAN x0_2:
                // Predict the clean image (x0_2) at the future timestep (prev_t) using epsilon_2
                let x0_2 = (x_prev_pred - epsilon_2.clone().mul_scalar(beta_prev.sqrt()))
                    .div_scalar(alpha_prev.sqrt());
                    
                // 5. AVERAGING:
                // Average the predicted clean states (x0) and noise directions (epsilon)
                let x0_avg = (x0_1 + x0_2).mul_scalar(0.5);
                let epsilon_avg = (epsilon_1 + epsilon_2).mul_scalar(0.5);
                
                // 6. FINAL INTEGRATION:
                // Integrate using the averaged clean state and noise scaled by the target timestep factors
                x_t = x0_avg.mul_scalar(alpha_prev.sqrt()) + epsilon_avg.mul_scalar(beta_prev.sqrt());
            } else {
                // Standard 1st-Order DDIM step (Euler method equivalent)
                x_t = scheduler.step(x_t, epsilon_1, t, prev_t);
            }
        }
        
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
        
        // Generate with 5 steps for test performance (DDPM mode)
        let history = generate_image_steps(&model, &device, 3, 5, "linear", "ddim", "noise");
        assert_eq!(history.len(), 6); // 1 initial noise state + 5 denoising steps
        assert_eq!(history[0].len(), 784);

        // Generate with 5 steps for test performance (Flow Matching mode)
        let history_fm = generate_image_steps(&model, &device, 3, 5, "linear", "ddim", "velocity");
        assert_eq!(history_fm.len(), 6);
        assert_eq!(history_fm[0].len(), 784);
    }
}

use burn::prelude::*;

/// A DDIM (Denoising Diffusion Implicit Models) / DDPM scheduler.
/// This scheduler is backend-agnostic and manages both the forward noise schedules
/// and the reverse sampling steps.
pub struct DDIMScheduler {
    pub num_train_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
}

impl DDIMScheduler {
    pub fn new(num_train_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut betas = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            betas.push(beta_start + t * (beta_end - beta_start));
        }
        
        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut current_prod = 1.0;
        for &a in &alphas {
            current_prod *= a;
            alphas_cumprod.push(current_prod);
        }
        
        Self {
            num_train_timesteps,
            betas,
            alphas,
            alphas_cumprod,
        }
    }
    
    /// Get the cumulative product of alpha (alpha_bar) at a specific step.
    pub fn get_alpha_cumprod(&self, step: usize) -> f32 {
        self.alphas_cumprod[step]
    }
    
    /// Forward process: Add noise to images at given timesteps
    /// x0 shape: [B, C, H, W]
    /// noise shape: [B, C, H, W]
    /// timesteps shape: [B] containing index of steps (0..num_train_timesteps)
    pub fn add_noise<B: Backend>(&self, x0: Tensor<B, 4>, noise: Tensor<B, 4>, timesteps: Tensor<B, 1, Int>) -> Tensor<B, 4> {
        let device = x0.device();
        let batch_size = x0.shape().dims[0];
        
        // Extract alpha_cumprod for the batch of timesteps
        let steps_vec = timesteps.into_data().into_vec::<i64>().unwrap();
        
        let mut sqrt_alphas_cumprod_vec = Vec::with_capacity(batch_size);
        let mut sqrt_one_minus_alphas_cumprod_vec = Vec::with_capacity(batch_size);
        
        for step in steps_vec {
            let step_idx = step.clamp(0, (self.num_train_timesteps - 1) as i64) as usize;
            let alpha_cp = self.alphas_cumprod[step_idx];
            sqrt_alphas_cumprod_vec.push(alpha_cp.sqrt());
            sqrt_one_minus_alphas_cumprod_vec.push((1.0 - alpha_cp).sqrt());
        }
        
        // Create 4D tensors of shape [B, 1, 1, 1] to scale the images and noise
        let scale_x = Tensor::<B, 1>::from_floats(sqrt_alphas_cumprod_vec.as_slice(), &device)
            .reshape([batch_size, 1, 1, 1]);
        let scale_n = Tensor::<B, 1>::from_floats(sqrt_one_minus_alphas_cumprod_vec.as_slice(), &device)
            .reshape([batch_size, 1, 1, 1]);
            
        x0 * scale_x + noise * scale_n
    }

    /// Predict the sample at the previous timestep using the DDIM formula.
    /// x_t: current sample [B, C, H, W]
    /// model_output: predicted noise [B, C, H, W]
    /// timestep: current step index (0..num_train_timesteps)
    /// prev_timestep: previous step index (can be smaller by step_ratio)
    pub fn step<B: Backend>(
        &self,
        x_t: Tensor<B, 4>,
        model_output: Tensor<B, 4>,
        timestep: usize,
        prev_timestep: Option<usize>,
    ) -> Tensor<B, 4> {
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = match prev_timestep {
            Some(prev) => self.alphas_cumprod[prev],
            None => 1.0, // If t=0, previous is clean image (alpha_prod = 1.0)
        };
        
        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;
        
        // 1. Predict x0 (clean image) from current x_t and predicted noise
        // x0_pred = (x_t - sqrt(1 - alpha_bar_t) * model_output) / sqrt(alpha_bar_t)
        let x0_pred = (x_t.clone() - model_output.clone().mul_scalar(beta_prod_t.sqrt()))
            .div_scalar(alpha_prod_t.sqrt());
            
        // 2. Compute direction pointing to x_t
        // pred_dir = sqrt(1 - alpha_bar_t_prev) * model_output
        let pred_dir = model_output.mul_scalar(beta_prod_t_prev.sqrt());
        
        // 3. Compute x_{t-1} = sqrt(alpha_bar_t_prev) * x0_pred + pred_dir
        x0_pred.mul_scalar(alpha_prod_t_prev.sqrt()) + pred_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_add_noise() {
        let device = Default::default();
        let scheduler = DDIMScheduler::new(1000, 1e-4, 0.02);
        
        // Create dummy clean image (all 1.0s) of shape [1, 1, 28, 28]
        let x0 = Tensor::<NdArray, 4>::ones([1, 1, 28, 28], &device);
        let noise = Tensor::<NdArray, 4>::zeros([1, 1, 28, 28], &device); // No noise
        
        // Test step 0
        let t0 = Tensor::<NdArray, 1, Int>::from_ints([0], &device);
        let noisy0 = scheduler.add_noise(x0.clone(), noise.clone(), t0);
        let val0 = noisy0.into_data().into_vec::<f32>().unwrap()[0];
        // At t=0, alpha_cumprod is close to 1.0, so noisy image should be close to 1.0
        assert!((val0 - scheduler.alphas_cumprod[0].sqrt()).abs() < 1e-5);
        
        // Test step 999
        let t999 = Tensor::<NdArray, 1, Int>::from_ints([999], &device);
        let noisy999 = scheduler.add_noise(x0, noise, t999);
        let val999 = noisy999.into_data().into_vec::<f32>().unwrap()[0];
        // At t=999, alpha_cumprod is very small, so noisy image should be close to 0.0 (since noise is zero)
        assert!((val999 - scheduler.alphas_cumprod[999].sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_step() {
        let device = Default::default();
        let scheduler = DDIMScheduler::new(1000, 1e-4, 0.02);
        
        let x_t = Tensor::<NdArray, 4>::ones([1, 1, 28, 28], &device);
        let model_output = Tensor::<NdArray, 4>::ones([1, 1, 28, 28], &device) * 0.1;
        let prev = scheduler.step(x_t, model_output, 10, Some(9));
        let prev_vec = prev.into_data().into_vec::<f32>().unwrap();
        assert_eq!(prev_vec.len(), 784);
    }
}

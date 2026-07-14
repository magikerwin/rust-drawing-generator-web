use wasm_bindgen::prelude::*;
use burn::{
    backend::NdArray,
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use model_shared::{Model, DDIMScheduler};

const MODEL_VERSION: &str = include_str!("../weights-version.txt");

#[wasm_bindgen]
pub fn get_compiled_model_version() -> String {
    MODEL_VERSION.trim().to_string()
}

#[wasm_bindgen]
pub struct GeneratorWasm {
    model: Model<NdArray>,
    device: <NdArray as Backend>::Device,
    scheduler: DDIMScheduler,
    x_t: Tensor<NdArray, 4>,
    class_ids: Tensor<NdArray, 1, Int>,
    steps: Vec<usize>,
    current_step_idx: usize,
    sampler: String,
}

#[wasm_bindgen]
impl GeneratorWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], num_classes: usize, class_id: usize, num_steps: usize, schedule: String, sampler: String) -> Result<GeneratorWasm, JsValue> {
        console_error_panic_hook::set_once();
        let device = Default::default();
        
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(model_bytes.to_vec(), &device)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model weights: {:?}", e)))?;
            
        let model = Model::<NdArray>::new(&device, num_classes).load_record(record);
        let scheduler = DDIMScheduler::new(1000, 1e-4, 0.02);
        
        let x_t = Tensor::<NdArray, 4>::random(
            [1, 1, 28, 28],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let class_ids = Tensor::<NdArray, 1, Int>::from_ints([class_id as i32], &device);
        
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
            let rho: f32 = match schedule.as_str() {
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
        
        Ok(Self {
            model,
            device,
            scheduler,
            x_t,
            class_ids,
            steps,
            current_step_idx: 0,
            sampler,
        })
    }
    
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }
    
    pub fn current_step(&self) -> usize {
        self.current_step_idx
    }
    
    pub fn is_complete(&self) -> bool {
        self.current_step_idx >= self.steps.len()
    }
    
    pub fn get_current_pixels(&self) -> Result<Vec<f32>, JsValue> {
        let data = self.x_t.clone().into_data().into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&format!("Failed to extract tensor data: {:?}", e)))?;
        let pixels = data.into_iter()
            .map(|val| {
                let denorm = (val + 1.0) * 127.5;
                denorm.clamp(0.0, 255.0)
            })
            .collect();
        Ok(pixels)
    }
    
    pub fn step(&mut self) -> Result<Option<Vec<f32>>, JsValue> {
        if self.is_complete() {
            return Ok(None);
        }
        
        let t = self.steps[self.current_step_idx];
        let prev_t = if self.current_step_idx + 1 < self.steps.len() {
            Some(self.steps[self.current_step_idx + 1])
        } else {
            None
        };
        
        let timesteps = Tensor::<NdArray, 1>::from_floats([t as f32], &self.device);
        let epsilon_1 = self.model.unet.forward(self.x_t.clone(), timesteps, self.class_ids.clone());
        
        if self.sampler == "heun" && prev_t.is_some() {
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
            
            let alpha_t = self.scheduler.alphas_cumprod[t];
            let alpha_prev = self.scheduler.alphas_cumprod[prev_t_val];
            let beta_t = 1.0 - alpha_t;
            let beta_prev = 1.0 - alpha_prev;
            
            // 1. PREDICT CLEAN x0_1:
            // Predict the clean image (x0_1) from current state x_t and current noise (epsilon_1)
            let x0_1 = (self.x_t.clone() - epsilon_1.clone().mul_scalar(beta_t.sqrt()))
                .div_scalar(alpha_t.sqrt());
                
            // 2. PREDICTOR STEP (Euler Step to next state):
            // Project the state forward to an intermediate state (x_prev_pred) at the next timestep (prev_t)
            let x_prev_pred = x0_1.clone().mul_scalar(alpha_prev.sqrt()) + epsilon_1.clone().mul_scalar(beta_prev.sqrt());
            
            // 3. CORRECTOR STEP:
            // Evaluate the model's U-Net again at the estimated intermediate state (x_prev_pred)
            // at the future timestep (prev_t) to get the predicted future noise (epsilon_2).
            let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], &self.device);
            let epsilon_2 = self.model.unet.forward(x_prev_pred.clone(), prev_timesteps, self.class_ids.clone());
            
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
            self.x_t = x0_avg.mul_scalar(alpha_prev.sqrt()) + epsilon_avg.mul_scalar(beta_prev.sqrt());
        } else {
            // Standard 1st-Order DDIM step (Euler method equivalent)
            self.x_t = self.scheduler.step(self.x_t.clone(), epsilon_1, t, prev_t);
        }
        
        self.current_step_idx += 1;
        
        let pixels = self.get_current_pixels()?;
        Ok(Some(pixels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    use burn::record::Recorder;

    #[wasm_bindgen_test]
    fn test_generator_wasm_creation() {
        let device = Default::default();
        let model = Model::<NdArray>::new(&device, 10);
        
        // Serialize the model's record in memory to get bytes
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(model.into_record(), ()).unwrap();
        
        // Create GeneratorWasm from those bytes
        let mut generator = GeneratorWasm::new(&bytes, 10, 3, 5, "linear".to_string(), "ddim".to_string()).unwrap();
        assert_eq!(generator.total_steps(), 5);
        assert_eq!(generator.current_step(), 0);
        assert!(!generator.is_complete());
        
        // Get initial pixels
        let initial_pixels = generator.get_current_pixels().unwrap();
        assert_eq!(initial_pixels.len(), 784);
        
        // Step once
        let step_pixels = generator.step().unwrap().unwrap();
        assert_eq!(step_pixels.len(), 784);
        assert_eq!(generator.current_step(), 1);
    }
}

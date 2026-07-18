use wasm_bindgen::prelude::*;
use burn::{
    backend::NdArray,
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use model_shared::{Model, DDIMScheduler};

// Embed model version tag compiled into WASM metadata
const MODEL_VERSION: &str = include_str!("../../docs/weights-version.txt");

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
    prediction_type: String, // "noise" or "velocity"
    cfg_scale: f32,
}

#[wasm_bindgen]
impl GeneratorWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_bytes: &[u8],
        num_classes: usize,
        class_id: usize,
        num_steps: usize,
        schedule: String,
        sampler: String,
        prediction_type: Option<String>,
        cfg_scale: Option<f32>,
    ) -> Result<GeneratorWasm, JsValue> {
        console_error_panic_hook::set_once();
        let device = Default::default();
        
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(model_bytes.to_vec(), &device)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model weights: {:?}", e)))?;
            
        let model = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Model::<NdArray>::new(&device, num_classes).load_record(record)
        }))
        .map_err(|err| {
            let msg = if let Some(s) = err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = err.downcast_ref::<String>() {
                s.to_string()
            } else {
                "Shape/structural mismatch between the compiled WASM model and the downloaded weights file.".to_string()
            };
            JsValue::from_str(&format!(
                "Model load failed: {}. ACTION required: Please retrain the dataset using the latest model architecture, convert and publish the weights locally using 'cargo run --release --bin publish_weights -- <version>', and commit & push the updated version files to GitHub to deploy matching assets.",
                msg
            ))
        })?;
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
        
        let prediction_type_str = prediction_type.unwrap_or_else(|| "noise".to_string());
        let prediction_type = if prediction_type_str == "velocity" {
            "velocity".to_string()
        } else {
            "noise".to_string()
        };
        
        let cfg_scale_val = cfg_scale.unwrap_or(1.0);
        
        Ok(Self {
            model,
            device,
            scheduler,
            x_t,
            class_ids,
            steps,
            current_step_idx: 0,
            sampler,
            prediction_type,
            cfg_scale: cfg_scale_val,
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
        
        // Define guided forward pass closure
        let unet_forward = |x: Tensor<NdArray, 4>, t_tensor: Tensor<NdArray, 1>| -> Tensor<NdArray, 4> {
            if self.cfg_scale == 1.0 {
                self.model.unet.forward(x, t_tensor, self.class_ids.clone())
            } else {
                let class_ids_uncond = Tensor::<NdArray, 1, Int>::from_ints([self.model.unet.num_classes as i32], &self.device);
                let out_cond = self.model.unet.forward(x.clone(), t_tensor.clone(), self.class_ids.clone());
                let out_uncond = self.model.unet.forward(x, t_tensor, class_ids_uncond);
                out_uncond.clone() + (out_cond - out_uncond).mul_scalar(self.cfg_scale)
            }
        };
        
        let timesteps = Tensor::<NdArray, 1>::from_floats([t as f32], &self.device);
        let out_1 = unet_forward(self.x_t.clone(), timesteps);
        
        if self.prediction_type == "velocity" {
            // --- EDUCATIONAL: FLOW MATCHING REVERSE ODE SOLVERS ---
            // In Flow Matching, the network outputs velocity v_t = x_1 - x_0.
            // Integrate backward from t=1.0 down to t=0.0.
            let t_scaled = t as f32 / 1000.0;
            let prev_t_val = prev_t.unwrap_or(0);
            let prev_t_scaled = prev_t_val as f32 / 1000.0;
            let dt = t_scaled - prev_t_scaled;
            
            if self.sampler == "heun" && prev_t.is_some() {
                // Heun's 2nd-order predictor-corrector method:
                let x_prev_pred = self.x_t.clone() - out_1.clone().mul_scalar(dt);
                let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], &self.device);
                let out_2 = unet_forward(x_prev_pred, prev_timesteps);
                let v_avg = (out_1 + out_2).mul_scalar(0.5);
                self.x_t = self.x_t.clone() - v_avg.mul_scalar(dt);
            } else {
                // Euler 1st-order method:
                self.x_t = self.x_t.clone() - out_1.mul_scalar(dt);
            }
        } else {
            // --- DDPM / DDIM REVERSE PROCESS ---
            let epsilon_1 = out_1;
            
            if self.sampler == "heun" && prev_t.is_some() {
                let prev_t_val = prev_t.unwrap();
                
                // --- EDUCATIONAL: HEUN'S 2ND-ORDER SAMPLER (PREDICTOR-CORRECTOR METHOD) ---
                // While DDIM works as a 1st-order solver (Euler method) that takes steps along the 
                // initial derivative (epsilon_1), Heun's method computes a 2nd-order correction step 
                // to drastically decrease numerical approximation errors along the ODE trajectory.
                
                let alpha_t = self.scheduler.alphas_cumprod[t];
                let alpha_prev = self.scheduler.alphas_cumprod[prev_t_val];
                let beta_t = 1.0 - alpha_t;
                let beta_prev = 1.0 - alpha_prev;
                
                // 1. PREDICT CLEAN x0_1:
                let x0_1 = (self.x_t.clone() - epsilon_1.clone().mul_scalar(beta_t.sqrt()))
                    .div_scalar(alpha_t.sqrt());
                    
                // 2. PREDICTOR STEP (Euler Step to next state):
                let x_prev_pred = x0_1.clone().mul_scalar(alpha_prev.sqrt()) + epsilon_1.clone().mul_scalar(beta_prev.sqrt());
                
                // 3. CORRECTOR STEP:
                let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], &self.device);
                let epsilon_2 = unet_forward(x_prev_pred.clone(), prev_timesteps);
                
                // 4. PREDICT CLEAN x0_2:
                let x0_2 = (x_prev_pred - epsilon_2.clone().mul_scalar(beta_prev.sqrt()))
                    .div_scalar(alpha_prev.sqrt());
                    
                // 5. AVERAGING:
                let x0_avg = (x0_1 + x0_2).mul_scalar(0.5);
                let epsilon_avg = (epsilon_1 + epsilon_2).mul_scalar(0.5);
                
                // 6. FINAL INTEGRATION:
                self.x_t = x0_avg.mul_scalar(alpha_prev.sqrt()) + epsilon_avg.mul_scalar(beta_prev.sqrt());
            } else {
                // Standard 1st-Order DDIM step (Euler method equivalent)
                self.x_t = self.scheduler.step(self.x_t.clone(), epsilon_1, t, prev_t);
            }
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
        
        // Create GeneratorWasm from those bytes (DDPM mode)
        let mut generator = GeneratorWasm::new(&bytes, 10, 3, 5, "linear".to_string(), "ddim".to_string(), Some("noise".to_string())).unwrap();
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

        // Create GeneratorWasm from those bytes (Flow Matching mode)
        let mut generator_fm = GeneratorWasm::new(&bytes, 10, 3, 5, "linear".to_string(), "ddim".to_string(), Some("velocity".to_string())).unwrap();
        assert_eq!(generator_fm.total_steps(), 5);
        assert_eq!(generator_fm.current_step(), 0);
        
        let step_pixels_fm = generator_fm.step().unwrap().unwrap();
        assert_eq!(step_pixels_fm.len(), 784);
        assert_eq!(generator_fm.current_step(), 1);
    }
}

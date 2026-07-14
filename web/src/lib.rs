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
            
            // 1. Predictor step
            let x_prev_pred = self.scheduler.step(self.x_t.clone(), epsilon_1.clone(), t, prev_t);
            
            // 2. Corrector step (evaluate noise at predicted next state)
            let prev_timesteps = Tensor::<NdArray, 1>::from_floats([prev_t_val as f32], &self.device);
            let epsilon_2 = self.model.unet.forward(x_prev_pred, prev_timesteps, self.class_ids.clone());
            
            // Average predicted noise
            let epsilon_avg = (epsilon_1 + epsilon_2).mul_scalar(0.5);
            
            // Recompute final step using averaged noise
            self.x_t = self.scheduler.step(self.x_t.clone(), epsilon_avg, t, prev_t);
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

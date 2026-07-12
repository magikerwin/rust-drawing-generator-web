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
}

#[wasm_bindgen]
impl GeneratorWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], num_classes: usize, class_id: usize, num_steps: usize) -> Result<GeneratorWasm, JsValue> {
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
        let step_ratio = 1000 / num_steps;
        for i in (0..num_steps).rev() {
            steps.push(i * step_ratio);
        }
        
        Ok(Self {
            model,
            device,
            scheduler,
            x_t,
            class_ids,
            steps,
            current_step_idx: 0,
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
        
        let predicted_noise = self.model.unet.forward(self.x_t.clone(), timesteps, self.class_ids.clone());
        self.x_t = self.scheduler.step(self.x_t.clone(), predicted_noise, t, prev_t);
        
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
        let mut generator = GeneratorWasm::new(&bytes, 10, 3, 5).unwrap();
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

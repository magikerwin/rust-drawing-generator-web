# MNIST Classifier with Burn (Rust)

A simple MNIST handwritten digit classifier built using the **Burn** deep learning framework in Rust.

## Commands

### Run Training
To start the training loop (using the CPU ndarray backend):
```powershell
cargo run --release
```
*Note: Always run with the `--release` flag so that compiling takes advantage of compiler optimizations for tensor math.*

### Run Tests
To run the unit tests (verifies model shapes, data pipeline, and training config):
```powershell
cargo test
```

### Run Inference (Predict Digits)
Once the model is trained and saved in `./target/mnist-model/`, you can load it to run inference on test images by passing the `--predict` flag:
```powershell
cargo run --release -- --predict
```

When run, the program will render a sample test digit in ASCII Art on your console and output the model's prediction:
```text
Loading model for inference...

Input Image:
      ######                
      ################      
      ################      
           ###########      
                  ####      
                 ####       
                 ####       
                ####        
                ####        
               ####         
               ###          
              ####          
             ####           
            #####           
            ####            
           #####            
           ####             
          #####             
          #####             
          ####              
                            
Target Label (Ground Truth): 7
Model Prediction           : 7
```

## Expected Results

After running the 5 epochs of training, the MLP model converges to the following performance on the MNIST validation/test dataset:

*   **Validation Accuracy**: `95.53%`
*   **Validation Loss**: `0.209`

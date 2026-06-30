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

## Expected Results

After running the 5 epochs of training, the MLP model converges to the following performance on the MNIST validation/test dataset:

*   **Validation Accuracy**: `95.53%`
*   **Validation Loss**: `0.209`

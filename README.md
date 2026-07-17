# 🎨 Rust Drawing Generator — Burn (Rust)

> An interactive drawing generator utilizing a Denoising Diffusion Probabilistic Model (DDPM/DDIM) built with the [Burn](https://burn.dev/) deep learning framework in Rust. Generate handwritten digits, letters, or doodles right in your browser!
>
> 🚀 **[Try the Live WebAssembly Demo!](https://magikerwin.github.io/rust-drawing-generator-web/)**

![image](assets/web_demo_mnist.webp)

---

## 📑 Table of Contents

- [Features](#-features)
- [Model Architecture](#️-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Train the Model](#train-the-model)
  - [Run Tests](#run-tests)
  - [CLI Generation](#cli-generation)
  - [Interactive Web Server (Axum backend)](#interactive-web-server-axum-backend)
  - [Client-Side WebAssembly App (WASM)](#client-side-webassembly-app-wasm)
- [Quick, Draw! Generation Details](#-quick-draw-generation-details)
- [References](#-references)
- [License](#-license)

---

## ✨ Features

- **Conditional U-Net Architecture** — Sinusoidal time embedding module, class conditioning embedding module, skip connections, and residual blocks.
- **DDIM & Heun Scheduler Denoising** — Accelerated reverse sampling supporting both **DDIM (1st-Order)** and **Heun (2nd-Order)** samplers, with customizable **Polynomial Spacing Schedules** (exponents 1.0–7.0) to achieve superior drawing quality in as few as 5–10 steps.
- **Progressive Denoising Animation** — Visual progressive rendering showing the drawing emerge frame-by-frame from pure Gaussian noise.
- **Dual Inference Modes** — Server-side Axum streaming via Server-Sent Events (SSE) and client-side WebAssembly local browser execution.
- **Fully in Rust** — Training, scheduler, inference engine, and web frontend in a unified workspace.

---

## 🏗️ Model Architecture

The generator relies on a lightweight Denoising Diffusion Probabilistic Model (DDPM/DDIM) with a conditional U-Net structure:

```
Inputs: Latent State x_t [B×1×28×28], Timestep t [B], Class ID c [B]
  → Time Embedding: Sinusoidal Positional Encoding (32) + MLP (32→128)
  → Class Embedding: Embedding Lookup (32) + Linear Projection (32→128)
  → Merged Embedding: Addition of Time & Class Embeddings [B×128]
  → U-Net Encoder:
      → Stem: Conv2d(1→32 channels) [B×32×28×28]
      → Down Block 1: UNetBlock + Time/Class Injection [B×32×28×28]
      → Downsample 1: Conv2d(32→64 channels, Stride 2) [B×64×14×14]
      → Down Block 2: UNetBlock + Time/Class Injection [B×64×14×14]
      → Downsample 2: Conv2d(64→128 channels, Stride 2) [B×128×7×7]
  → Bottleneck Middle Block: UNetBlock + Time/Class Injection [B×128×7×7]
  → U-Net Decoder:
      → Upsample 1: ConvTranspose2d(128→64 channels, Stride 2) [B×64×14×14]
      → Up Block 1: Concatenate(Upsample 1, Skip 2) → UNetBlock(128→64 channels) [B×64×14×14]
      → Upsample 2: ConvTranspose2d(64→32 channels, Stride 2) [B×32×28×28]
      → Up Block 2: Concatenate(Upsample 2, Skip 1) → UNetBlock(64→32 channels) [B×32×28×28]
      → Output Layer: Conv2d(32→1 channels) [B×1×28×28]
```

---

## 📁 Project Structure

```
rust-drawing-generator/
├── model_shared/           # Shared library workspace crate
│   ├── src/lib.rs          # Model architecture & DDIMScheduler definition
│   ├── src/unet.rs         # Conditional U-Net blocks and modules
│   └── src/scheduler.rs    # DDIM forward process and reverse sampling math
├── web/                    # Rust WASM crate (wasm-pack entry point)
│   └── src/lib.rs          # Stateful GeneratorWasm wrapper exposing .step()
├── src/                    # Training, CLI generation, & serving (Burn backend)
│   ├── main.rs             # CLI router & Axum API serve endpoint
│   ├── model.rs            # Re-exports shared Model wrapper
│   ├── training.rs         # Autodiff training loop and MSE loss wrapper
│   ├── inference.rs        # Iterative DDIM/Heun progressive sampling & ASCII art
│   ├── data.rs             # Noise collator & normalized dataset batcher
│   ├── emnist.rs           # EMNIST classes mapping helper
│   ├── quickdraw.rs        # QuickDraw dataset loading and class mappings
│   └── bin/                # Executable runner scripts
│       ├── build_web.rs    # Script to build, optimize, and bundle WASM
│       ├── convert.rs      # Utility to convert trained weights to binary format
│       └── publish_weights.rs # Utility to bundle and publish weights to release targets
├── docs/                   # Static web frontend (served by GitHub Pages)
│   ├── index.html          # Drawing generator UI with Developer Console
│   └── pkg/                # Compiled WASM output (gitignored, built by CI)
└── assets/                 # README showcase demo assets
```

---

## 🚀 Getting Started

### Train the Model

By default, training runs on the CPU (`NdArray` backend):

```sh
cargo run --release
```

To train on your GPU (`Wgpu` backend):

```sh
cargo run --release -- --gpu
```

To specify the number of training epochs (defaults to 5):

```sh
cargo run --release -- --epochs 25
```

To specify the training learning rate (defaults to 2e-4):

```sh
cargo run --release -- --lr 0.0003
```

To train on the **EMNIST Letters** dataset (26 classes):

```sh
cargo run --release -- --dataset emnist --gpu
```

To train on the Google **Quick, Draw!** dataset (25 classes):

```sh
cargo run --release -- --dataset quickdraw --gpu
```

> **Dataset Cache:** Dataset files are downloaded once and cached at `target/emnist_dataset/` and `target/quickdraw_dataset/`.

---

### Run Tests

Run the mathematical tests verifying the forward scheduling, time embeddings, and U-Net blocks:

```sh
cargo test
```

---

### CLI Generation

Generate drawings from random Gaussian noise and watch the progressive ASCII rendering directly in your terminal:

```sh
# Generate MNIST digits
cargo run --release -- --predict

# Generate EMNIST Letters
cargo run --release -- --predict --dataset emnist

# Generate Quick, Draw! doodles
cargo run --release -- --predict --dataset quickdraw
```

<details>
<summary>📝 Example Terminal Output (MNIST digit 4 progress)</summary>

```text
Loading model for generation (dataset: mnist)...
Generating drawing for class: '4' (class ID: 4) using 20 DDIM steps...

Generated Output:
                            
                            
               ..           
               ##.          
              .###          
              ####          
             #####          
            .#####          
            #####           
           .####.           
           #####            
          ######            
         #######            
         ####.##            
        ##### ##            
       .#### .##            
       ####  .##            
      ######.###.           
      ##########.           
      ##########            
          .  ##             
             ##.            
             ##.            
             ##.            
                            
                            

Generation complete!
```

</details>

---

### Interactive Web Server (Axum backend)

Start the browser-based generator UI backed by the Rust Axum server streaming denoising steps via Server-Sent Events (SSE):

- **MNIST Digits (default)**:
  ```sh
  cargo run --release -- --serve
  ```
  Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to generate digits (0–9).

- **EMNIST Letters**:
  ```sh
  cargo run --release -- --serve --dataset emnist
  ```
  Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to generate letters (A–Z).

- **Quick, Draw! Doodles**:
  ```sh
  cargo run --release -- --serve --dataset quickdraw
  ```
  Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to generate doodles (25 classes).

#### GET `/api/generate` SSE Stream

The Axum server exposes a GET endpoint `/api/generate` that streams the progressive denoising steps using Server-Sent Events (SSE):

- **Query Parameters**:
  - `class_id` (integer, required): The target class ID to generate.
  - `steps` (integer, optional, default: `20`): Number of denoising steps (clamped between `5` and `100`).
  - `schedule` (string, optional, default: `"linear"`): Denoising schedule type or power exponent (e.g. `"linear"`, `"quadratic"`, or a custom float string like `"3.0"`, `"7.0"`).
  - `sampler` (string, optional, default: `"ddim"`): Denoising sampler algorithm: `"ddim"` or `"heun"`.

---

### Client-Side WebAssembly App (WASM)

The models compile to WebAssembly for fully client-side generation. Model weights (`*-model.bin`) are downloaded on-demand in the browser and cached locally.

#### 1. Build the WASM bundle locally

Make sure you have trained the models first, then:

1. **Install wasm-pack**:
   ```sh
   cargo install wasm-pack
   ```

2. **Build the WebAssembly module**:
   ```sh
   cargo run --bin build_web
   ```

3. **Install a local static file server**:
   ```sh
   cargo install basic-http-server
   ```

4. **Serve locally**:
   ```sh
   basic-http-server docs
   ```
   Navigate to **[http://localhost:4000](http://localhost:4000)**.

#### 2. Automatic Deployments & Release Management

The CI workflow (`.github/workflows/deploy.yml`) automatically builds and deploys to GitHub Pages on every push to `master`.

To update the model weights used by the CI runner:

1. Upload your local weights to a GitHub Release:
   ```sh
   cargo run --bin publish_weights
   # or with custom tag
   cargo run --bin publish_weights -- v2.0.0
   ```

2. Commit and push the updated version files:
   ```sh
   git add web/weights-version.txt docs/weights-version.txt
   git commit -m "Update model weights version"
   git push origin master
   ```

---

## 🎨 Quick, Draw! Generation Details

This project supports doodle generation using a subset of the **[Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**.

### Selected Categories (25 classes)

We train on a curated subset of **25 doodle classes**:

* **Nature / Weather:** `sun`, `moon`, `star`, `tree`, `flower`
* **Animals:** `cat`, `dog`, `fish`, `butterfly`
* **Common Objects:** `cup`, `key`, `umbrella`, `hat`, `clock`, `envelope`, `toothbrush`
* **Structures / Vehicles:** `house`, `car`
* **Shapes:** `circle`, `triangle`, `square`, `smiley face`
* **Clothing:** `pants`, `t-shirt`
* **Food:** `apple`

---

## 📚 References

- [Burn — Deep Learning Framework for Rust](https://burn.dev/)
- [The Burn Book](https://burn.dev/book/)
- [wasm-pack — Rust WebAssembly Packager](https://rustwasm.github.io/wasm-pack/)
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
- [EMNIST Dataset (NIST Special Database 19)](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
- [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

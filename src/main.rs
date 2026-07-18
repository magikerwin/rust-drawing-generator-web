mod model;
mod data;
mod training;
mod inference;
mod quickdraw;
mod emnist;

use burn::{
    backend::{Autodiff, NdArray},
    data::dataset::{vision::MnistDataset, Dataset},
    optim::AdamConfig,
};
use crate::training::{train, TrainingConfig};
use crate::inference::{load_model, generate_image_steps, render_ascii};

use axum::{
    extract::{State, Query},
    response::Html,
    routing::get,
    Json, Router,
};
use axum::response::sse::{Event, Sse};
use futures_util::stream::{self, Stream};
use std::sync::Arc;
use tower_http::services::ServeDir;
use std::convert::Infallible;
use std::time::Duration;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize)]
struct AppConfig {
    dataset: String,
    classes: Vec<String>,
    version: String,
    compiled_version: String,
    prediction_type: String,
}

/// Shared state across the web server requests
#[derive(Clone)]
struct AppState {
    model: Arc<std::sync::Mutex<crate::model::Model<NdArray>>>,
    device: <NdArray as burn::prelude::Backend>::Device,
    config: AppConfig,
}

/// Query parameters for generation requests
#[derive(Deserialize)]
struct GenerateQuery {
    class_id: usize,
    steps: Option<usize>,
    schedule: Option<String>,
    sampler: Option<String>,
}

/// SSE step payload structure
#[derive(Serialize)]
struct SseStepPayload {
    step: usize,
    total_steps: usize,
    pixels: Vec<f32>,
}

/// Helper function to load the model's prediction target configuration from config.json.
/// If config.json does not exist or lacks the parameter, defaults to DDPM/DDIM ("noise").
fn get_prediction_type_from_config(artifact_dir: &str) -> String {
    let config_path = format!("{artifact_dir}/config.json");
    if let Ok(content) = std::fs::read_to_string(config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(pred_type) = json.get("prediction_type").and_then(|v| v.as_str()) {
                return pred_type.to_string();
            }
        }
    }
    "noise".to_string()
}

/// Handler that serves the HTML drawing canvas frontend page
async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../docs/index.html"))
}

/// Handler that serves the model weights version dynamically
async fn weights_version_handler() -> Result<String, axum::http::StatusCode> {
    tokio::fs::read_to_string("docs/weights-version.txt")
        .await
        .map_err(|_| axum::http::StatusCode::NOT_FOUND)
}

/// Handler that streams intermediate generation steps via Server-Sent Events (SSE)
async fn generate_handler(
    State(state): State<AppState>,
    Query(query): Query<GenerateQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let steps = query.steps.unwrap_or(16).clamp(1, 128);
    let schedule = query.schedule.as_deref().unwrap_or("linear");
    let sampler = query.sampler.as_deref().unwrap_or("ddim");
    
    // Perform progressive generation on the CPU using the model's prediction type
    let history = {
        let model = state.model.lock().unwrap();
        generate_image_steps(
            &model,
            &state.device,
            query.class_id,
            steps,
            schedule,
            sampler,
            &state.config.prediction_type,
        )
    };

    let total = history.len();
    
    // Create an asynchronous stream yielding each frame with an artificial 35ms delay
    let stream = stream::unfold((history, 0), move |(history, idx)| async move {
        if idx >= history.len() {
            None
        } else {
            let pixels = history[idx].clone();
            let payload = SseStepPayload {
                step: idx,
                total_steps: total - 1,
                pixels,
            };
            let json = serde_json::to_string(&payload).unwrap();
            let event = Event::default().data(json);
            
            // Add a small delay for smooth visual transition on the client canvas
            tokio::time::sleep(Duration::from_millis(35)).await;
            
            Some((Ok(event), (history, idx + 1)))
        }
    });

    Sse::new(stream)
}

/// Handler that serves the dataset and labels configuration
async fn config_handler(
    State(state): State<AppState>,
) -> Json<AppConfig> {
    Json(state.config.clone())
}

#[tokio::main]
async fn main() {
    // CLI argument parsing
    let args: Vec<String> = std::env::args().collect();
    let run_inference = args.contains(&"--predict".to_string());
    let run_server = args.contains(&"--serve".to_string());
    let run_gpu = args.contains(&"--gpu".to_string());

    let dataset_arg = args.iter()
        .position(|arg| arg == "--dataset")
        .and_then(|pos| args.get(pos + 1))
        .map(|s| s.as_str())
        .unwrap_or("mnist");

    let prediction_type_arg = args.iter()
        .position(|arg| arg == "--prediction-type")
        .and_then(|pos| args.get(pos + 1))
        .map(|s| s.as_str())
        .unwrap_or("noise");

    let epochs_arg = args.iter()
        .position(|arg| arg == "--epochs")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse::<usize>().ok());

    let lr_arg = args.iter()
        .position(|arg| arg == "--lr")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse::<f64>().ok());

    let num_classes = if dataset_arg == "quickdraw" {
        quickdraw::QUICKDRAW_CLASSES.len()
    } else if dataset_arg == "emnist" {
        emnist::EMNIST_CLASSES.len()
    } else {
        10
    };

    let artifact_dir = if dataset_arg == "quickdraw" {
        "./target/quickdraw-model"
    } else if dataset_arg == "emnist" {
        "./target/emnist-model"
    } else {
        "./target/mnist-model"
    };

    if run_server {
        // ==========================================
        // BRANCH A: RUN INTERACTIVE WEB SERVER
        // ==========================================
        println!("Loading model for web server (dataset: {})...", dataset_arg);
        let device = Default::default(); // NdArray CPU device
        let model = Arc::new(std::sync::Mutex::new(load_model(artifact_dir, &device, num_classes)));
        
        let classes = if dataset_arg == "quickdraw" {
            quickdraw::QUICKDRAW_CLASSES.iter().map(|&s| s.to_string()).collect()
        } else if dataset_arg == "emnist" {
            emnist::EMNIST_CLASSES.iter().map(|&s| s.to_string()).collect()
        } else {
            (0..10).map(|i| i.to_string()).collect()
        };
        
        let version = std::fs::read_to_string("docs/weights-version.txt")
            .unwrap_or_else(|_| "unknown".to_string())
            .trim()
            .to_string();

        let compiled_version = std::fs::read_to_string("web/weights-version.txt")
            .unwrap_or_else(|_| "unknown".to_string())
            .trim()
            .to_string();

        let prediction_type = get_prediction_type_from_config(artifact_dir);
        println!("Model target prediction type loaded: {}", prediction_type);

        let config = AppConfig {
            dataset: dataset_arg.to_string(),
            classes,
            version: version.clone(),
            compiled_version,
            prediction_type,
        };
        let state = AppState { model, device, config };

        // Construct Axum application routing
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/weights-version.txt", get(weights_version_handler))
            .route("/api/generate", get(generate_handler))
            .route("/api/config", get(config_handler))
            .fallback_service(ServeDir::new("docs"))
            .with_state(state);

        // Bind TCP listener to port 3000
        let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
            .await
            .unwrap();
        println!("\n==================================================");
        println!("   Burn Drawing Generator Web Server Running!");
        println!("   Open your browser to: http://127.0.0.1:3000");
        println!("==================================================\n");

        axum::serve(listener, app).await.unwrap();

    } else if run_inference {
        // ==========================================
        // BRANCH B: RUN CLI PREDICTION (GENERATION)
        // ==========================================
        println!("Loading model for generation (dataset: {})...", dataset_arg);
        let device = Default::default();
        let model = load_model(artifact_dir, &device, num_classes);

        // Choose a random class to generate
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let class_id = (nanos % num_classes as u128) as usize;

        let class_name = if dataset_arg == "quickdraw" {
            quickdraw::QUICKDRAW_CLASSES[class_id].to_string()
        } else if dataset_arg == "emnist" {
            emnist::EMNIST_CLASSES[class_id].to_string()
        } else {
            class_id.to_string()
        };

        println!("Generating drawing for class: '{}' (class ID: {}) using 20 steps...", class_name, class_id);
        
        let prediction_type = get_prediction_type_from_config(artifact_dir);
        let history = generate_image_steps(&model, &device, class_id, 20, "linear", "ddim", &prediction_type);
        
        // Render final drawing as ASCII art
        println!("\nGenerated Output:");
        render_ascii(history.last().unwrap());
        println!("\nGeneration complete!");

    } else {
        // ==========================================
        // BRANCH C: RUN TRAINING LOOP
        // ==========================================
        let mut config = TrainingConfig::new(AdamConfig::new());
        config.prediction_type = prediction_type_arg.to_string();
        if let Some(epochs) = epochs_arg {
            config.num_epochs = epochs;
        }
        if let Some(lr) = lr_arg {
            config.learning_rate = lr;
        }

        if dataset_arg == "quickdraw" {
            let train_dataset = quickdraw::QuickDrawDataset::new(true, quickdraw::TRAIN_SAMPLES_PER_CLASS);
            let valid_dataset = quickdraw::QuickDrawDataset::new(false, quickdraw::VAL_SAMPLES_PER_CLASS);

            if run_gpu {
                println!("Starting Quick, Draw! training on GPU (WGPU backend)...");
                train::<Autodiff<burn::backend::Wgpu>, _, _>(
                    artifact_dir,
                    config,
                    burn::backend::wgpu::WgpuDevice::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    true, // allow_horizontal_flip
                );
            } else {
                println!("Starting Quick, Draw! training on CPU (NdArray backend)...");
                train::<Autodiff<NdArray>, _, _>(
                    artifact_dir,
                    config,
                    Default::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    true, // allow_horizontal_flip
                );
            }
        } else if dataset_arg == "emnist" {
            use burn::data::dataset::InMemDataset;

            println!("Loading EMNIST Letters dataset into memory...");
            let emnist_train = emnist::EmnistDataset::new(true);
            let train_len = emnist_train.len();
            let mut train_items = Vec::with_capacity(train_len);
            println!("  Loading {} train items...", train_len);
            for i in 0..train_len {
                if let Some(item) = emnist_train.get(i) {
                    train_items.push(item);
                }
                if (i + 1) % 30000 == 0 {
                    println!("    Parsed {}/{} train items...", i + 1, train_len);
                }
            }
            let train_dataset = InMemDataset::new(train_items);

            let emnist_test = emnist::EmnistDataset::new(false);
            let test_len = emnist_test.len();
            let mut valid_items = Vec::with_capacity(test_len);
            println!("  Loading {} validation items...", test_len);
            for i in 0..test_len {
                if let Some(item) = emnist_test.get(i) {
                    valid_items.push(item);
                }
                if (i + 1) % 5000 == 0 {
                    println!("    Parsed {}/{} validation items...", i + 1, test_len);
                }
            }
            let valid_dataset = InMemDataset::new(valid_items);

            if run_gpu {
                println!("Starting EMNIST Letters training on GPU (WGPU backend)...");
                train::<Autodiff<burn::backend::Wgpu>, _, _>(
                    artifact_dir,
                    config,
                    burn::backend::wgpu::WgpuDevice::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    false, // allow_horizontal_flip
                );
            } else {
                println!("Starting EMNIST Letters training on CPU (NdArray backend)...");
                train::<Autodiff<NdArray>, _, _>(
                    artifact_dir,
                    config,
                    Default::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    false, // allow_horizontal_flip
                );
            }
        } else {
            use burn::data::dataset::InMemDataset;
            
            println!("Loading MNIST dataset into memory...");
            
            let mnist_train = MnistDataset::train();
            let train_len = mnist_train.len();
            let mut train_items = Vec::with_capacity(train_len);
            println!("  Loading {} train items...", train_len);
            for i in 0..train_len {
                if let Some(item) = mnist_train.get(i) {
                    train_items.push(item);
                }
                if (i + 1) % 15000 == 0 {
                    println!("    Parsed {}/{} train items...", i + 1, train_len);
                }
            }
            let train_dataset = InMemDataset::new(train_items);

            let mnist_test = MnistDataset::test();
            let test_len = mnist_test.len();
            let mut valid_items = Vec::with_capacity(test_len);
            println!("  Loading {} validation items...", test_len);
            for i in 0..test_len {
                if let Some(item) = mnist_test.get(i) {
                    valid_items.push(item);
                }
                if (i + 1) % 2500 == 0 {
                    println!("    Parsed {}/{} validation items...", i + 1, test_len);
                }
            }
            let valid_dataset = InMemDataset::new(valid_items);

            if run_gpu {
                println!("Starting MNIST training on GPU (WGPU backend)...");
                train::<Autodiff<burn::backend::Wgpu>, _, _>(
                    artifact_dir,
                    config,
                    burn::backend::wgpu::WgpuDevice::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    false, // allow_horizontal_flip
                );
            } else {
                println!("Starting MNIST training on CPU (NdArray backend)...");
                train::<Autodiff<NdArray>, _, _>(
                    artifact_dir,
                    config,
                    Default::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                    false, // allow_horizontal_flip
                );
            }
        }
        println!("Training finished! Model saved successfully to {}", artifact_dir);
    }
}

use std::fs;
use std::path::Path;
use std::process::Command;

/// This build script runs before compiling the `mnist_web` library.
/// It ensures that the model weights are present in `OUT_DIR` so that
/// they can be embedded into the WebAssembly binary via `include_bytes!`.
///
/// This avoids committing large binary files to Git, while maintaining a smooth
/// offline local development cycle and automated GitHub Actions builds.
fn main() {
    // Tell Cargo to re-run this script if build.rs or weights-version.txt changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=weights-version.txt");

    // Read the target weights release version from weights-version.txt
    let version = fs::read_to_string("weights-version.txt")
        .unwrap_or_else(|_| {
            panic!(
                "Failed to read weights-version.txt. Run ./publish-weights.ps1 to sync the release version, or update web/weights-version.txt manually."
            )
        })
        .trim()
        .to_string();

    if version.is_empty() {
        panic!("weights-version.txt is empty. Run ./publish-weights.ps1 to populate it.");
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_mnist = Path::new(&out_dir).join("mnist-model.bin");
    let dest_qd = Path::new(&out_dir).join("quickdraw-model.bin");

    // Relative path to local training build output directories
    let local_mnist = Path::new("../target/mnist-model/model.bin");
    let local_qd = Path::new("../target/quickdraw-model/model.bin");

    // --- Resolve MNIST Model Weights ---
    if local_mnist.exists() {
        // Crucial: Tell Cargo to re-run this script if the local model file is updated
        println!("cargo:rerun-if-changed={}", local_mnist.display());

        // Local Retraining Loop: copy fresh weights directly from the target folder
        println!("cargo:warning=Using local MNIST weights from target/");
        fs::copy(local_mnist, &dest_mnist).unwrap();
    } else {
        // CI/New Developer Fallback: download stable weights from GitHub Releases
        println!("cargo:warning=Local MNIST weights not found in target/, downloading from GitHub Releases...");
        download_file(
            &format!("https://github.com/magikerwin/rust-burn-classifier-web/releases/download/{}/mnist-model.bin", version),
            &dest_mnist,
        ).unwrap();
    }

    // --- Resolve Quick Draw! Model Weights ---
    if local_qd.exists() {
        // Crucial: Tell Cargo to re-run this script if the local model file is updated
        println!("cargo:rerun-if-changed={}", local_qd.display());

        // Local Retraining Loop: copy fresh weights directly from target folder
        println!("cargo:warning=Using local Quick Draw weights from target/");
        fs::copy(local_qd, &dest_qd).unwrap();
    } else {
        // CI/New Developer Fallback: download stable weights from GitHub Releases
        println!("cargo:warning=Local Quick Draw weights not found in target/, downloading from GitHub Releases...");
        download_file(
            &format!("https://github.com/magikerwin/rust-burn-classifier-web/releases/download/{}/quickdraw-model.bin", version),
            &dest_qd,
        ).unwrap();
    }
}


/// Downloads a file using `curl`.
/// Using std::process::Command calling `curl` keeps build dependencies light
/// and avoids compiled dependency bloat, as curl is pre-installed on Windows, macOS, and Linux.
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let dest_str = dest.to_str().ok_or("Invalid destination path")?;
    let status = Command::new("curl")
        .args(&["-L", "-f", "-o", dest_str, url])
        .status()
        .map_err(|e| format!("Failed to run curl: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("curl failed with status: {:?}", status))
    }
}

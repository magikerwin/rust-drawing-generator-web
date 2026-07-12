use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get the version from CLI arguments (after '--'), defaulting to v1.0.0
    let args: Vec<String> = env::args().collect();
    let version = if args.len() > 1 {
        &args[1]
    } else {
        "v1.0.0"
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_path = PathBuf::from(manifest_dir);

    // Verify GitHub CLI (gh) is installed
    match Command::new("gh").arg("--version").output() {
        Ok(_) => {}
        Err(_) => {
            eprintln!("Error: GitHub CLI (gh) is not installed or not in PATH.");
            eprintln!("Please install it and log in using 'gh auth login' before running this tool.");
            std::process::exit(1);
        }
    }

    // Check if local model weights exist in target directory
    let mnist_weights = root_path.join("target/mnist-model/model.bin");
    let qd_weights = root_path.join("target/quickdraw-model/model.bin");
    let emnist_weights = root_path.join("target/emnist-model/model.bin");

    let mut present_datasets = Vec::new();
    if mnist_weights.exists() {
        present_datasets.push(("mnist-model.bin", mnist_weights, root_path.join("docs/mnist-model.bin")));
    } else {
        println!("Info: Local MNIST weights not found. Skipping.");
    }
    if qd_weights.exists() {
        present_datasets.push(("quickdraw-model.bin", qd_weights, root_path.join("docs/quickdraw-model.bin")));
    } else {
        println!("Info: Local Quick Draw weights not found. Skipping.");
    }
    if emnist_weights.exists() {
        present_datasets.push(("emnist-model.bin", emnist_weights, root_path.join("docs/emnist-model.bin")));
    } else {
        println!("Info: Local EMNIST weights not found. Skipping.");
    }

    if present_datasets.is_empty() {
        eprintln!("Error: No trained model weights found in target/ directories.");
        eprintln!("Please train at least one dataset first, e.g.: cargo run --release -- --dataset mnist");
        std::process::exit(1);
    }

    println!("Ensuring GitHub Release '{}' exists...", version);
    // Try creating the release. If it already exists, gh CLI will report a warning but continue safely.
    let notes = format!(
        "Pre-trained model weights for offline WebAssembly inference ({})",
        version
    );
    let _ = Command::new("gh")
        .args([
            "release",
            "create",
            version,
            "--title",
            version,
            "--notes",
            &notes,
        ])
        .status();

    println!("Preparing model binaries for upload and local dev...");
    let mut files_to_upload = Vec::new();

    for (filename, src_path, docs_path) in &present_datasets {
        let temp_dest = root_path.join(filename);
        if let Err(e) = fs::copy(src_path, &temp_dest) {
            eprintln!("Failed to copy to {:?}: {}", temp_dest, e);
            for f in &files_to_upload {
                let _ = fs::remove_file(root_path.join(f));
            }
            std::process::exit(1);
        }
        if let Err(e) = fs::copy(src_path, docs_path) {
            eprintln!("Failed to copy to {:?}: {}", docs_path, e);
            let _ = fs::remove_file(&temp_dest);
            for f in &files_to_upload {
                let _ = fs::remove_file(root_path.join(f));
            }
            std::process::exit(1);
        }
        files_to_upload.push(filename.to_string());
    }

    println!(
        "Uploading model weights to GitHub Release {} (overwriting previous assets)...",
        version
    );
    let mut upload_args = vec!["release", "upload", version];
    for f in &files_to_upload {
        upload_args.push(f);
    }
    upload_args.push("--clobber");

    let upload_status = Command::new("gh")
        .args(&upload_args)
        .current_dir(&root_path)
        .status();

    // Clean up temporary files
    println!("Cleaning up temporary files...");
    for f in &files_to_upload {
        let _ = fs::remove_file(root_path.join(f));
    }

    match upload_status {
        Ok(status) if status.success() => {
            // Success
        }
        _ => {
            eprintln!("Error: Failed to upload release assets to GitHub.");
            std::process::exit(1);
        }
    }

    println!("Updating weights version files...");
    let version_files = [
        root_path.join("web/weights-version.txt"),
        root_path.join("docs/weights-version.txt"),
    ];

    for file_path in &version_files {
        if let Err(e) = fs::write(file_path, version) {
            eprintln!("Warning: Failed to write to {:?}: {}", file_path, e);
        } else {
            if let Some(name) = file_path.file_name() {
                println!("Updated {:?} to {}.", name, version);
            }
        }
    }

    println!(
        "Success! Model weights uploaded successfully to GitHub Release {}.",
        version
    );
    println!("Remember to commit and push the updated version files so CI and the web UI use the new release.");
}

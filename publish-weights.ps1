param (
    [string]$Version = "v1.0.0"
)

# Verify GitHub CLI (gh) is installed
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "GitHub CLI (gh) is not installed. Please install it and log in using 'gh auth login' before running this script."
    exit 1
}

# Check if local model weights exist in target directory
$mnistWeights = "target/mnist-model/model.bin"
$qdWeights = "target/quickdraw-model/model.bin"

if (-not (Test-Path $mnistWeights)) {
    Write-Error "Local MNIST weights not found at '$mnistWeights'. Please run training first using: cargo run --release -- --dataset mnist"
    exit 1
}
if (-not (Test-Path $qdWeights)) {
    Write-Error "Local Quick Draw weights not found at '$qdWeights'. Please run training first using: cargo run --release -- --dataset quickdraw"
    exit 1
}

Write-Host "Ensuring GitHub Release '$Version' exists..." -ForegroundColor Cyan
# Try creating the release. If it already exists, gh CLI will report a warning but continue safely
gh release create $Version --title "$Version" --notes "Pre-trained model weights for offline WebAssembly inference ($Version)" 2>$null

Write-Host "Preparing model binaries for upload..." -ForegroundColor Cyan
Copy-Item $mnistWeights "mnist-model.bin"
Copy-Item $qdWeights "quickdraw-model.bin"

Write-Host "Uploading model weights to GitHub Release $Version (overwriting previous assets)..." -ForegroundColor Cyan
gh release upload $Version "mnist-model.bin" "quickdraw-model.bin" --clobber

Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
Remove-Item "mnist-model.bin" -Force
Remove-Item "quickdraw-model.bin" -Force

Write-Host "Success! Model weights uploaded successfully to GitHub Release $Version." -ForegroundColor Green

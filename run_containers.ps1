# =============================================================================
# run_containers.ps1 — Windows PowerShell script to launch containerized services
# Detects podman compose or docker compose automatically.
# =============================================================================

Write-Host "Checking container runtime..." -ForegroundColor Cyan

$hasPodman = Get-Command "podman" -ErrorAction SilentlyContinue
$hasDocker = Get-Command "docker" -ErrorAction SilentlyContinue

if ($hasPodman) {
    Write-Host "Using Podman Compose..." -ForegroundColor Green
    podman compose -f compose.yaml up -d --build
} elseif ($hasDocker) {
    Write-Host "Using Docker Compose..." -ForegroundColor Green
    docker compose -f compose.yaml up -d --build
} else {
    Write-Error "Neither Podman nor Docker was found on PATH. Please install Podman or Docker Desktop."
    exit 1
}

Write-Host "`nServices launching:" -ForegroundColor Cyan
Write-Host "  - Django REST API:    http://localhost:8000/api/v1/health/"
Write-Host "  - Streamlit Frontend: http://localhost:8501"
Write-Host "  - Qdrant Vector DB:   http://localhost:6333/dashboard"
Write-Host "  - MongoDB:            mongodb://localhost:27017"

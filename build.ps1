#!/usr/bin/env pwsh

Write-Host "🐳 Building Docker Images" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Function to build and show size
function Build-Image {
    param(
        [string]$Mode,
        [string]$Tag,
        [string]$Description
    )
    
    Write-Host "📦 Building $Description..." -ForegroundColor Yellow
    
    $env:DOCKER_BUILDKIT = "1"
    $buildArgs = @("--build-arg", "BUILD_MODE=$Mode", "--tag", $Tag, ".")
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    docker build @buildArgs
    $stopwatch.Stop()
    
    if ($LASTEXITCODE -eq 0) {
        $size = docker images $Tag --format "{{.Size}}" | Select-Object -First 1
        $time = $stopwatch.Elapsed.ToString("mm\:ss")
        Write-Host "✅ Built $Tag" -ForegroundColor Green
        Write-Host "   Size: $size | Time: $time" -ForegroundColor Gray
    } else {
        Write-Host "❌ Failed to build $Tag" -ForegroundColor Red
    }
    Write-Host ""
}

# Build both versions
Build-Image "lite" "moke-pitch:lite" "Lite Version (Core functionality, no heavy ML)"
Build-Image "full" "moke-pitch:full" "Full Version (All features including ML models)"

Write-Host "📊 Image Comparison:" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
docker images moke-pitch --format "table {{.Repository}}:{{.Tag}}`t{{.Size}}`t{{.CreatedAt}}"

Write-Host ""
Write-Host "🚀 Usage:" -ForegroundColor Cyan
Write-Host "=========" -ForegroundColor Cyan
Write-Host "Lite version:  docker run -p 8080:8080 moke-pitch:lite" -ForegroundColor White
Write-Host "Full version:  docker run -p 8080:8080 moke-pitch:full" -ForegroundColor White
Write-Host ""
Write-Host "💡 Recommendations:" -ForegroundColor Yellow
Write-Host "- Use 'lite' for faster deployment and smaller size" -ForegroundColor Gray
Write-Host "- Use 'full' for complete ML functionality" -ForegroundColor Gray
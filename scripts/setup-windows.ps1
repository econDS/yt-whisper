param(
    [string]$DataRoot = 'E:\yt-whisper',
    [string]$CondaExecutable = (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe')
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
foreach ($part in @('tmp', 'cache\pip', 'cache\conda-pkgs')) {
    New-Item -ItemType Directory -Path (Join-Path $DataRoot $part) -Force | Out-Null
}
$env:YTW_DATA_DIR = $DataRoot
$env:TEMP = Join-Path $DataRoot 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $DataRoot 'cache\pip'
$env:CONDA_PKGS_DIRS = Join-Path $DataRoot 'cache\conda-pkgs'
$prefix = Join-Path $DataRoot 'envs\py312'
$python = Join-Path $prefix 'python.exe'
if (!(Test-Path -LiteralPath $python)) {
    & $CondaExecutable create --prefix $prefix --override-channels -c conda-forge python=3.12 pip -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}
$env:Path = "$prefix;$prefix\Scripts;$prefix\Library\bin;" + $env:Path
Push-Location $repo
try {
    & $python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    $lock = Join-Path $repo 'requirements-windows-py312.lock.txt'
    if (Test-Path -LiteralPath $lock) {
        & $python -m pip install -c $lock -e '.[ui,thai,dev]'
    } else {
        & $python -m pip install -e '.[ui,thai,dev]'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Application installation failed.' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & $python -m yt_whisper.cli --doctor
    if ($LASTEXITCODE -ne 0) { throw 'Application preflight failed.' }
} finally {
    Pop-Location
}

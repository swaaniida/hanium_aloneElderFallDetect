$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

python -c "import pyrealsense2 as rs; devices=list(rs.context().query_devices()); print(f'RealSense devices: {len(devices)}'); [print(f'  {d.get_info(rs.camera_info.name)} ({d.get_info(rs.camera_info.serial_number)})') for d in devices]; raise SystemExit(0 if devices else 2)"

if ($LASTEXITCODE -ne 0) {
    Write-Error "RealSense camera not found. Connect it to USB 3.0 and run this script again."
    exit $LASTEXITCODE
}

python -m tsstg_pipeline.collect `
    --subject-id S001 `
    --location location_2 `
    --ncnn-threads 2

exit $LASTEXITCODE

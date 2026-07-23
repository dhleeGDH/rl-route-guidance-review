param(
  [string]$WorkDir, [int]$Obs, [string]$Out,
  [int]$Iters = 50, [int]$Batch = 320, [int]$Seed = 0,
  [int]$MaxLaunches = 120, [int]$LaunchTimeoutSec = 300
)
# Relaunch the chunked trainer until all iters are done (exit 42). Each launch runs one iter
# (~7 episodes, under the ~12-episode ray SUMO-restart crash threshold), saves, exits; the
# next launch restores that checkpoint. A native access violation sometimes leaves the python
# process HUNG (ray/TF cleanup deadlock) instead of exiting, so every launch is bounded by a
# hard timeout: past it the process tree is killed and the chunk is retried from the last
# checkpoint. Stray SUMO/python are reaped before every launch so leaked handles never pile up.
$py = "C:\xrv\Scripts\python.exe"
$script = "D:\review_paper\drl-rgs-review\experiments\xrouting_mc4\train_open.py"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:XR_WORKDIR = $WorkDir; $env:XR_OBS = "$Obs"; $env:XR_OUT = $Out
$env:XR_ITERS = "$Iters"; $env:XR_CHUNK = "1"; $env:XR_BATCH = "$Batch"; $env:XR_SEED = "$Seed"
$orch = Join-Path $Out "orchestrator.log"
$outLog = Join-Path $Out "launch_out.log"; $errLog = Join-Path $Out "launch_err.log"
Add-Content $orch ("=== START {0} WorkDir={1} Obs={2} Iters={3} Batch={4} ===" -f (Get-Date -Format 'HH:mm:ss'), $WorkDir, $Obs, $Iters, $Batch)
$stateFile = Join-Path $Out "state.json"
function Get-DoneIters {
  if (Test-Path $stateFile) { try { return (Get-Content $stateFile -Raw | ConvertFrom-Json).done_iters } catch { return -1 } }
  return 0
}
# Progress is judged from state.json (train_open.py writes done_iters on each saved chunk),
# NOT from the child exit code: Start-Process -PassThru + redirected stdio does not reliably
# surface ExitCode on this PS build, which previously mis-scored saved chunks as failures.
$consecFail = 0
for ($i = 1; $i -le $MaxLaunches; $i++) {
  Get-Process python, sumo, sumo-gui -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 500
  $prev = Get-DoneIters
  if ($prev -ge $Iters) { Add-Content $orch "COMPLETE (already at $prev)"; break }
  $p = Start-Process -FilePath $py -ArgumentList "`"$script`"" -PassThru -NoNewWindow `
       -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  $hung = $false
  if (-not $p.WaitForExit($LaunchTimeoutSec * 1000)) {
    $hung = $true
    try { $p | Stop-Process -Force -ErrorAction SilentlyContinue } catch {}
    Get-Process python, sumo, sumo-gui -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  }
  $cur = Get-DoneIters
  $line = (Get-Content $outLog -ErrorAction SilentlyContinue | Where-Object { $_ -match 'ITER \d|CHUNK DONE|RESTORED' } | Select-Object -Last 1)
  Add-Content $orch ("[{0}] launch {1} iters {2}->{3}{4} :: {5}" -f (Get-Date -Format 'HH:mm:ss'), $i, $prev, $cur, $(if($hung){" HUNG"}else{""}), $line)
  if ($cur -ge $Iters) { Add-Content $orch "COMPLETE"; break }
  elseif ($cur -gt $prev) { $consecFail = 0 }
  else { $consecFail++; if ($consecFail -ge 8) { Add-Content $orch "ABORT: 8 consecutive no-progress launches"; break } }
}
Get-Process python, sumo, sumo-gui -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Add-Content $orch ("=== END {0} ===" -f (Get-Date -Format 'HH:mm:ss'))

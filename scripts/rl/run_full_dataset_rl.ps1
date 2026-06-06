param(
    [string]$Timeseries = "results\rl_cidt_metrics_history\prepared_timeseries.csv",
    [string]$ResultsRoot = "results\rl_cidt_metrics_history",
    [int]$QEpisodes = 200,
    [int]$DqnTimesteps = 1000000
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$logDir = Join-Path $ResultsRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-LoggedStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogPath
    )

    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    & $python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    "[$(Get-Date -Format o)] END $Name exit=$exitCode" | Tee-Object -FilePath $LogPath -Append
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

Invoke-LoggedStep `
    -Name "full_heuristics_plus_q_learning" `
    -LogPath (Join-Path $logDir "all_methods_full_q.log") `
    -Arguments @(
        "scripts\rl\run_timeseries_experiments.py",
        "--timeseries", $Timeseries,
        "--scenario", "normal",
        "--job-count", "48",
        "--job-seeds", "42,43,44",
        "--include-q",
        "--q-episodes", "$QEpisodes",
        "--results-dir", (Join-Path $ResultsRoot "all_methods_full_q")
    )

Invoke-LoggedStep `
    -Name "full_timeseries_dqn" `
    -LogPath (Join-Path $logDir "timeseries_dqn_full.log") `
    -Arguments @(
        "scripts\rl\train_timeseries_dqn.py",
        "--timeseries", $Timeseries,
        "--scenario", "normal",
        "--job-count", "48",
        "--train-snapshots", "0",
        "--eval-snapshots", "0",
        "--eval-job-seeds", "42,43,44",
        "--timesteps", "$DqnTimesteps",
        "--device", "auto",
        "--results-dir", (Join-Path $ResultsRoot "timeseries_dqn_full_1m")
    )

"[$(Get-Date -Format o)] FULL DATASET RL RUN COMPLETE" | Tee-Object -FilePath (Join-Path $logDir "full_dataset_rl.done.log") -Append

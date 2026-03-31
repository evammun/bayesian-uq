# vast.ai Cheat Sheet

## Template settings
- **Image:** `vastai/llama-cpp:b8493-cuda-12.9`
- **Launch mode:** Interactive shell server, SSH (direct SSH checked)
- **On-start:** `bash /workspace/bayesian-uq/scripts/autorun.sh`
- **Filters:** `compute_cap>=610 cpu_arch in [amd64] cuda_max_good>=12.1 gpu_ram>=16000 gpu_total_ram>=32000 num_gpus in [1]`

## New rental (first time on a fresh instance)

```powershell
# 1. Rent interruptible, bid AT or ABOVE suggested price (too low = instant outbid)
# 2. Wait for green "Open" button
# 3. SSH in (get host:port from >_ icon)
ssh -p <PORT> root@<HOST> -L 8080:localhost:8080
# 4. In SSH:
mkdir -p /workspace/bayesian-uq
# 5. In a second PowerShell (same host:port!):
cd "C:\Users\evama\Dropbox\Family Room\Projects\bayesian-uq"
scp -P <PORT> -r src experiments scripts data results pyproject.toml root@<HOST>:/workspace/bayesian-uq/
# 6. Back in SSH:
cd /workspace/bayesian-uq && bash scripts/run_on_gpu.sh
# 7. Detach: Ctrl+B then D
# 8. Start local auto-fetch (leave running):
.\scripts\fetch_results.ps1 -SshHost root@<HOST> -SshPort <PORT> -Poll -IntervalSec 300
```

## If outbid

**Same instance restarts:** Do nothing. Results are still on disk. On-start resumes automatically.

**New instance (old one destroyed):** Your local `results/` folder has everything (the fetch script kept it synced). The scp command above includes `results`, so when you deploy to a new instance, it uploads your local results and the experiments resume from where they left off.

## Gotchas
- **"Permission denied (publickey)"** → re-add SSH key via key icon on instance, wait 15s
- **scp fails** → must use exact same host:port as SSH. If all else fails, paste code via SSH
- **Instance won't boot** → probably outbid at too-low price, not an on-start bug. Bid higher.
- **Dashboard shows wrong elapsed time** → timezone fix in `dashboard/app.py` line ~221, needs `utcfromtimestamp`
- **mkdir before scp** → always `mkdir -p /workspace/bayesian-uq` first or scp silently fails

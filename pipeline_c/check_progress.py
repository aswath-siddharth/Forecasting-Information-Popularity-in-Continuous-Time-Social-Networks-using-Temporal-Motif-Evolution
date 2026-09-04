import datetime
from pathlib import Path
import psutil

def main():
    print("=" * 60)
    print(" PIPELINE C: TRAINING STATUS & PROGRESS")
    print("=" * 60)

    # 1. Process Status
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'create_time']):
        try:
            cmd = " ".join(p.info['cmdline'] or [])
            if "python" in p.info['name'].lower() and "train" in cmd:
                procs.append(p)
        except Exception:
            pass

    if procs:
        print("\n[ACTIVE TRAINING PROCESSES]")
        for p in procs:
            runtime = datetime.timedelta(seconds=int(datetime.datetime.now().timestamp() - p.create_time()))
            ram_mb = p.memory_info().rss / (1024 ** 2)
            cpu_pct = p.cpu_percent(interval=0.2)
            print(f"  • PID: {p.pid:<6} | CPU: {cpu_pct:5.1f}% | RAM: {ram_mb:6.1f} MB | Runtime: {runtime}")
    else:
        print("\n[PROCESS STATUS] No active training process currently running.")

    # 2. Checkpoints & Outputs
    for dir_name in ["pipeline_c/outputs_full", "pipeline_c/outputs", "pipeline_c/outputs_test"]:
        output_dir = Path(dir_name)
        if output_dir.exists() and list(output_dir.glob("*")):
            print(f"\n[OUTPUT DIRECTORY: {output_dir}]")
            for f in sorted(output_dir.glob("*")):
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S")
                size_kb = f.stat().st_size / 1024
                print(f"  • {f.name:<30} | {size_kb:8.1f} KB | Updated: {mtime}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

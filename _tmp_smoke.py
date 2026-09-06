"""Full smoke run of calibrate_thresholds — accepts recommendation, then
immediately quits so config.py is NOT modified (simulates a 'n' response
to the confirm prompt)."""
import subprocess, sys
result = subprocess.run(
    [sys.executable, "scripts/calibrate_thresholds.py", "--split", "dev"],
    input="\nN\nq\n",   # accept rec, decline confirm, then quit
    text=True,
    capture_output=True,
    cwd=r"D:\VoxGuard",
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode not in (0, 1):  # 1 = user quit, also acceptable
    print("STDERR:", result.stderr[-500:])
    sys.exit(result.returncode)
print(f"\nExit code: {result.returncode} — OK")

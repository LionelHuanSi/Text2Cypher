# Kernel Kaggle (script, GPU T4 x2, Internet bật). Được scripts/kaggle_rq1.sh sinh ra từ run_template.py.
import glob
import subprocess
import sys

MODELS = __MODELS__
ARMS = __ARMS__

p = sorted(glob.glob("/kaggle/input/**/prompts.jsonl", recursive=True))
assert p, "không thấy prompts.jsonl trong /kaggle/input"
prompts = p[0]
gen = prompts.replace("prompts.jsonl", "gen_vllm.py")
subprocess.run("nvidia-smi", shell=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.6.6.post1"])
print("pip vllm exit", r.returncode, flush=True)
for m in MODELS:
    subprocess.run([sys.executable, gen, "--prompts", prompts, "--model", m, "--arms", *ARMS], check=False)
subprocess.run("cd /kaggle/working && tar czf out.tgz out && ls -la", shell=True)

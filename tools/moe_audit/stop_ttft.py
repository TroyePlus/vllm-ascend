"""Stop only descendants of the two launchers recorded by this benchmark."""

import signal
import sys
from pathlib import Path

import psutil

roots = [int(x) for x in Path(sys.argv[1]).read_text().split()]
processes = {}
for pid in roots:
    try:
        root = psutil.Process(pid)
        if "p-run_dp_template.sh" not in " ".join(root.cmdline()):
            raise RuntimeError(f"PID {pid} no longer identifies a benchmark launcher")
        for process in [root, *root.children(recursive=True)]:
            processes[process.pid] = process
    except psutil.NoSuchProcess:
        continue
for process in processes.values():
    try:
        process.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(list(processes.values()), timeout=10)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
print(f"Stopped benchmark tree rooted at {roots}; tracked {len(processes)} processes")

import os
import subprocess

folder = r"./examples"
example_files = [
    "eit_dynamic_bp.py",
    "eit_dynamic_jac.py",
    "eit_static_jac.py",
    "eit_dynamic_greit.py",
    "fem_forward2d.py",
    "fem_forward3d.py",
]

for ex in example_files:
    path = os.path.join(folder, ex)
    cmd = f"python {path}"
    print(f"runs >> {cmd}")
    subprocess.call(cmd, shell=True)

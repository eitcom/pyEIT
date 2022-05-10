import os
import subprocess

folder = r"./examples"
example = [
    "eit_dynamic_bp.py",
    "eit_dynamic_greit.py",
    "eit_dynamic_jac.py",
    "eit_dynamic_jac3d.py",
    "eit_dynamic_stack.py",
    "eit_dynamic_svd.py",
    "eit_sensitivity2d.py",
    "eit_static_GN_3D.py",
    "eit_static_jac.py",
    "fem_forward2d.py",
    "fem_forward3d.py",
    "mesh_distmesh2d.py",
    "mesh_distmesh3d.py",
    "mesh_intro2d.py",
    "mesh_multi_shell.py",
    "paper_eit2016b.py",
    "softx/figure01.py",
    "softx/figure02.py",
    "softx/figure02b.py",
    "softx/figure03.py",
]
list_ex = ""
index = {}
for i, file in enumerate(example):
    list_ex = f"{list_ex}Example #{i}: {file}\r\n"
    index[f"{i}"] = i


def run():
    ans = input(f"List of all examples:\r\n{list_ex} Run all examples? (y)/n or #: ")
    all = ans in ["Y", "y"]

    if not all and ans in list(index.keys()):
        _run_ex(example[index[ans]])
        return

    for ex in example:
        next = True
        if not all:
            ans = input(f"Run example '{ex}'? (y)/n:")
            next = ans not in ["N", "n"]
        if not next:
            continue
        _run_ex(ex)


def _run_ex(ex_):
    path = os.path.join(folder, ex_)
    cmd = f"python {path}"
    print(f"runs >> {cmd}")
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    """"""
    run()

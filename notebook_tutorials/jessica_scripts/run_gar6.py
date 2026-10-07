#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path

# ===========================================================================
# Settings
# ===========================================================================
PAIR_NUM = 2

BASE = Path(__file__).resolve().parent

DIR_2D = BASE / "gar6more2d"
DIR_3D = BASE / "gar6more3d"

BUILD_2D = DIR_2D / "build"
BUILD_3D = DIR_3D / "build"

OUTPUT_DIR = BASE / "results_gar6"

CASES = {
    1: {"name": "case_small_2d", "work": DIR_2D, "build": BUILD_2D,
        "executable": "Gar6more2D.out",
        "dat": "Gar6more2D-case_small_2d_fluid.dat",
        "input": "Gar6more2D.dat", "kind": "fluid"},
    2: {"name": "case_small_2d", "work": DIR_2D, "build": BUILD_2D,
        "executable": "Gar6more2D.out",
        "dat": "Gar6more2D-case_small_2d_solid.dat",
        "input": "Gar6more2D.dat", "kind": "solid"},
    3: {"name": "case_large_2d", "work": DIR_2D, "build": BUILD_2D,
        "executable": "Gar6more2D.out",
        "dat": "Gar6more2D-case_large_2d_fluid.dat",
        "input": "Gar6more2D.dat", "kind": "fluid"},
    4: {"name": "case_large_2d", "work": DIR_2D, "build": BUILD_2D,
        "executable": "Gar6more2D.out",
        "dat": "Gar6more2D-case_large_2d_solid.dat",
        "input": "Gar6more2D.dat", "kind": "solid"},
    5: {"name": "case_3d", "work": DIR_3D, "build": BUILD_3D,
        "executable": "Gar6more3D.out",
        "dat": "Gar6more3D-case_3d.dat",
        "input": "Gar6more3D.dat", "kind": "fluid"},
    6: {"name": "case_3d", "work": DIR_3D, "build": BUILD_3D,
        "executable": "Gar6more3D.out",
        "dat": "Gar6more3D-case_3d_solid.dat",
        "input": "Gar6more3D.dat", "kind": "solid"},
}

PAIRS = {
    1: (1, 2),
    2: (3, 4),
    3: (5, 6),
}

OUTPUT_FILES = ["Ux.dat", "Uy.dat", "P.dat"]

# ===========================================================================

def run_case(case_num: int) -> None:
    case = CASES[case_num]
    work = case["work"]
    build = case["build"]

    output_path = OUTPUT_DIR / case["name"] / case["kind"]
    output_path.mkdir(parents=True, exist_ok=True)

    dat_src = work / case["dat"]
    dat_dst = work / case["input"]
    exe = build / case["executable"]

    if not dat_src.exists():
        raise FileNotFoundError(f"[case {case_num}] .dat não encontrado: {dat_src}")
    if not exe.exists():
        raise FileNotFoundError(f"[case {case_num}] executável não encontrado: {exe}")

    print(f"\n================ CASE {case_num}: {case['name']} ({case['kind']}) ================")

    for f in OUTPUT_FILES:
        stale = work / f
        if stale.exists():
            print(f"  [clean] removendo sobra antiga: {stale.name} "
                  f"({stale.stat().st_size} bytes)")
            stale.unlink()

    shutil.copy(dat_src, dat_dst)
    print(f"  [copy]  {dat_src.name} -> {dat_dst.name}  "
          f"({dat_dst.stat().st_size} bytes)")

    print(f"  [run]   {exe}  (cwd={work})")
    subprocess.run(str(exe), shell=True, check=True, cwd=work)

    produced = [f for f in OUTPUT_FILES if (work / f).exists()]
    if not produced:
        raise RuntimeError(
            f"[case {case_num}] Executável rodou mas nenhuma saída "
            f"(Ux/Uy/P.dat) foi gerada em {work}. Verifique o .dat e o cwd."
        )
    print(f"  [check] saídas geradas: "
          f"{[(f, (work/f).stat().st_size) for f in produced]}")

    for f in produced:
        shutil.move(str(work / f), output_path / f)
        print(f"  [move]  {f} -> {output_path / f}")

    print(f"  ✓ Case {case_num} finalizado: {output_path}")


def main() -> None:
    pair = PAIRS[PAIR_NUM]
    print(f"=== Rodando par {PAIR_NUM}: cases {pair} ===")
    for case_num in pair:
        run_case(case_num)


if __name__ == "__main__":
    main()
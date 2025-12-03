import argparse
import subprocess
import shutil
import time
from pathlib import Path

SOLVER_BINARY = Path("./solver/build/kissat")
CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")

def decompress_if_xz(xz_file: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if xz_file.suffix == ".xz":
        temp = dest_dir / xz_file.name
        out = dest_dir / xz_file.with_suffix("").name
        if not out.exists():
            shutil.copy(xz_file, temp)
            subprocess.run(["xz", "-dkf", str(temp)], check=True)
        return out
    return xz_file

def run_solver(cnf_file: Path, extra_args: list[str], timeout: int = 5) -> tuple[str, float]:
    cmd = [str(SOLVER_BINARY.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    start = time.time()
    try:
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    elapsed = time.time() - start
    if r.stderr:
        print("stderr:", r.stderr.strip())
    return (r.stdout.strip() if r.stdout else "NO_OUTPUT"), elapsed

def resolve_path(path_arg: Path, default_dir: Path) -> Path:
    """Resuelve la ruta: si no existe, busca en default_dir"""
    if path_arg.exists():
        return path_arg
    alt = default_dir / path_arg.name
    if alt.exists():
        return alt
    return path_arg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cnf", type=Path, help="Archivo .cnf a probar")
    p.add_argument("--backbone", type=Path, default=None, help="Archivo backbone (opcional)")
    p.add_argument("--timeout", type=int, default=5, help="Timeout por ejecución (s)")
    p.add_argument("--neuroback_cfd", type=float, default=0.9, help="Umbral de confianza (default 0.9)")
    args = p.parse_args()

    if not SOLVER_BINARY.exists():
        print("ERROR: No se encontró el binario:", SOLVER_BINARY)
        return

    cnf = resolve_path(args.cnf, CNF_DIR)
    if not cnf.exists():
        print("ERROR: No se encontró el CNF:", cnf)
        return

    backbone = None
    if args.backbone:
        backbone = resolve_path(args.backbone, BACKBONE_DIR)
        if not backbone.exists():
            print("WARNING: No se encontró el backbone:", args.backbone)
            backbone = None
        else:
            backbone = decompress_if_xz(backbone, cnf.parent)

    print(f"\n{'='*60}")
    print(f"Prueba CNF: {cnf.name}")
    print(f"neuroback_cfd: {args.neuroback_cfd}")
    print(f"{'='*60}\n")

    # --- VERSIÓN 1: NeuroBack Normal ---
    if backbone:
        out, t = run_solver(cnf, [
            f"--backbonefile={str(backbone.resolve())}",
            "--neural_backbone_initial",
            f"--neuroback_cfd={args.neuroback_cfd}",
            "--stable=2"
        ], timeout=args.timeout)
        print(f"1. NeuroBack Normal        -> {out:15s} ({t:.2f}s)")
    else:
        print(f"1. NeuroBack Normal        -> SKIPPED (no backbone)")

    # --- VERSIÓN 2: NeuroBack Modificada ---
    if backbone:
        out, t = run_solver(cnf, [
            f"--backbonefile={str(backbone.resolve())}",
            "--neural_backbone_initial",
            "--neural_backbone_modified",
            f"--neuroback_cfd={args.neuroback_cfd}",
            "--stable=2"
        ], timeout=args.timeout)
        print(f"2. NeuroBack Modificada    -> {out:15s} ({t:.2f}s)")
    else:
        print(f"2. NeuroBack Modificada    -> SKIPPED (no backbone)")

    # --- VERSIÓN 3: Default Kissat ---
    out, t = run_solver(cnf, [], timeout=args.timeout)
    print(f"3. Default Kissat          -> {out:15s} ({t:.2f}s)")

    # --- VERSIÓN 4: Random Phase ---
    out, t = run_solver(cnf, [
        "--random_phase_initial",
        "--stable=0",
        f"--time={args.timeout}"
    ], timeout=args.timeout + 1)
    print(f"4. Random Phase Initial    -> {out:15s} ({t:.2f}s)")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
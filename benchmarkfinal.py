import argparse
import subprocess
import shutil
import time
import os
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN (Rutas fijas para evitar problemas)
# ==============================================================================
SOLVER_PATH = Path("./solver/build/kissat")
CNF_PATH = Path("./sym_data/cnf/test/3_102_423.cnf")

# Usamos el archivo que ya tienes y sabes que funciona (el Ground Truth)
BACKBONE_PATH = Path("./sym_data/backbone/test/3_102_423.cnf.backbone")
# ==============================================================================

def decompress_if_needed(file_path: Path) -> Path:
    if not file_path.exists():
        print(f"⚠️  ADVERTENCIA: El archivo no existe: {file_path}")
        return file_path

    if file_path.suffix == ".xz":
        target_path = file_path.with_suffix("")
        if target_path.exists(): return target_path
        print(f"   📦 Descomprimiendo {file_path.name}...")
        try:
            subprocess.run(["xz", "-dkk", str(file_path)], check=True)
        except Exception as e:
            print(f"   ❌ Error descomprimiendo: {e}")
        return target_path
    return file_path

def run_solver(cnf_file: Path, extra_args: list[str], timeout: int = 10) -> tuple[str, float]:
    cmd = [str(SOLVER_PATH.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    start = time.time()
    try:
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        out = r.stdout.strip() if r.stdout else ""
        err = r.stderr.strip() if r.stderr else ""
        
        if "SATISFIABLE" in out and "UNSATISFIABLE" not in out:
            res_str = "SATISFIABLE"
        elif "UNSATISFIABLE" in out:
            res_str = "UNSATISFIABLE"
        else:
            if err: res_str = f"ERROR: {err.splitlines()[0]}"
            else: res_str = f"UNKNOWN (Code {r.returncode})"
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", 0.0
    
    elapsed = time.time() - start
    return res_str, elapsed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=20, help="Timeout segundos")
    p.add_argument("--cfd", type=float, default=0.9, help="Confianza")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"BENCHMARK FINAL: TESTING NEW TARGET STRATEGY")
    print(f"{'='*70}")
    
    if not SOLVER_PATH.exists():
        print("❌ Falta el solver (haz make)"); return

    final_cnf = decompress_if_needed(CNF_PATH)
    final_backbone = BACKBONE_PATH
    if not final_backbone.exists():
        possible_xz = final_backbone.with_suffix(final_backbone.suffix + ".xz")
        if possible_xz.exists(): final_backbone = decompress_if_needed(possible_xz)
    
    backbone_ok = final_backbone.exists()

    print(f"CNF      : {final_cnf.name}")
    print(f"Backbone : {final_backbone.name if backbone_ok else 'NO ENCONTRADO'}")
    print("-" * 70)

    results = []

    # 1. NeuroBack ORIGINAL (Initial)
    if backbone_ok:
        res, t = run_solver(final_cnf, [
            f"--backbonefile={final_backbone.resolve()}",
            "--neural_backbone_initial",
            f"--neuroback_cfd={args.cfd}"
        ], timeout=args.timeout)
        results.append(f"1. NB Initial (Normal)    : {res:15s} ({t:.4f}s)")
    else: results.append("1. NB Initial             : SKIPPED")

    # 2. NeuroBack TARGET (¡TU NUEVA ESTRATEGIA!)
    # Nota: Usamos --stable=2 para potenciar el efecto del target
    if backbone_ok:
        res, t = run_solver(final_cnf, [
            f"--backbonefile={final_backbone.resolve()}",
            "--neural_backbone_target",
            "--stable=2", 
            f"--neuroback_cfd={args.cfd}"
        ], timeout=args.timeout)
        results.append(f"2. NB Target (Nueva)      : {res:15s} ({t:.4f}s)")
    else: results.append("2. NB Target              : SKIPPED")

    # 3. Default Kissat
    res, t = run_solver(final_cnf, [], timeout=args.timeout)
    results.append(f"3. Default Kissat         : {res:15s} ({t:.4f}s)")

    print("\nResultados Finales:")
    for r in results: print(r)
    print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
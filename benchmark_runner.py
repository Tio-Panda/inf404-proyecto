import argparse
import subprocess
import shutil
import time
import os
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN HARDCODEADA (¡Edita esto si tus archivos cambian!)
# ==============================================================================

# 1. Ruta al ejecutable de Kissat
SOLVER_PATH = Path("./solver/build/kissat")

# 2. Ruta al archivo CNF que quieres probar
CNF_PATH = Path("./sym_data/cnf/test/3_102_423.cnf")

# 3. Ruta al archivo Backbone
# OPCIÓN A: Usar el probabilistico (Recomendado para ver si tu heurística funciona)
BACKBONE_PATH = Path("./sym_data/backbone/test/3_102_423.cnf.backbone")

# OPCIÓN B: Usar el original (Descomenta esto si quieres usar el real)
# BACKBONE_PATH = Path("./sym_data/backbone/test/3_102_423.cnf.backbone")

# ==============================================================================

def decompress_if_needed(file_path: Path) -> Path:
    """
    Si el archivo indicado termina en .xz, lo descomprime en su misma carpeta
    quitándole la extensión .xz. Retorna la ruta del archivo descomprimido.
    """
    if not file_path.exists():
        print(f"⚠️  ADVERTENCIA: El archivo no existe: {file_path}")
        return file_path

    if file_path.suffix == ".xz":
        target_path = file_path.with_suffix("") # Quita .xz
        
        # Si ya existe el descomprimido, lo usamos directo
        if target_path.exists():
            return target_path

        print(f"   📦 Descomprimiendo {file_path.name}...")
        try:
            # Copia temporal para no dañar el original si falla
            subprocess.run(["xz", "-dkk", str(file_path)], check=True)
            # xz -k mantiene el original y crea el descomprimido
        except Exception as e:
            print(f"   ❌ Error descomprimiendo: {e}")
        
        return target_path
    
    return file_path

def run_solver(cnf_file: Path, extra_args: list[str], timeout: int = 10) -> tuple[str, float]:
    # Construir comando
    cmd = [str(SOLVER_PATH.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    
    start = time.time()
    try:
        # Ejecutar solver
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        out = r.stdout.strip() if r.stdout else ""
        
        # Parsear resultado rápido
        if "SATISFIABLE" in out and "UNSATISFIABLE" not in out:
            res_str = "SATISFIABLE"
        elif "UNSATISFIABLE" in out:
            res_str = "UNSATISFIABLE"
        else:
            res_str = "UNKNOWN/ERR"
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    except Exception as e:
        return f"ERROR: {e}", 0.0
    
    elapsed = time.time() - start
    return res_str, elapsed

def main():
    # Argumentos opcionales solo para timeout y cfd, los archivos son fijos arriba
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=15, help="Timeout segundos")
    p.add_argument("--cfd", type=float, default=0.9, help="Confianza NeuroBack")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"BENCHMARK HARDCODED")
    print(f"{'='*70}")
    
    # 1. Validar Solver
    if not SOLVER_PATH.exists():
        print(f"❌ ERROR CRÍTICO: No encuentro el solver en: {SOLVER_PATH}")
        return

    # 2. Preparar CNF
    final_cnf = decompress_if_needed(CNF_PATH)
    if not final_cnf.exists():
        print(f"❌ ERROR CRÍTICO: No encuentro el CNF en: {final_cnf}")
        return

    # 3. Preparar Backbone
    # Primero chequeamos si existe el que pusiste en la variable.
    # Si no, intentamos ver si existe con .xz
    final_backbone = BACKBONE_PATH
    if not final_backbone.exists():
        possible_xz = final_backbone.with_suffix(final_backbone.suffix + ".xz")
        if possible_xz.exists():
            final_backbone = decompress_if_needed(possible_xz)
    
    backbone_ok = final_backbone.exists()

    print(f"Solver   : {SOLVER_PATH}")
    print(f"CNF      : {final_cnf.name}")
    print(f"Backbone : {final_backbone.name if backbone_ok else 'NO ENCONTRADO'}")
    print(f"Timeout  : {args.timeout}s  |  CFD: {args.cfd}")
    print("-" * 70)

    results = []

    # --- EJECUCIÓN 1: NeuroBack NORMAL ---
    if backbone_ok:
        res, t = run_solver(final_cnf, [
            f"--backbonefile={final_backbone.resolve()}",
            "--neural_backbone_initial",
            f"--neuroback_cfd={args.cfd}",
            "--stable=2"
        ], timeout=args.timeout)
        results.append(f"1. NeuroBack Normal       : {res:15s} ({t:.4f}s)")
    else:
        results.append("1. NeuroBack Normal       : SKIPPED (No backbone)")

    # --- EJECUCIÓN 2: NeuroBack MODIFICADA (Tu Heurística) ---
    if backbone_ok:
        res, t = run_solver(final_cnf, [
            f"--backbonefile={final_backbone.resolve()}",
            "--neural_backbone_initial",
            "--neural_backbone_modified",  # <--- Activa tu lógica rebelde
            f"--neuroback_cfd={args.cfd}",
            "--stable=2"
        ], timeout=args.timeout)
        results.append(f"2. NeuroBack Modificada   : {res:15s} ({t:.4f}s)")
    else:
        results.append("2. NeuroBack Modificada   : SKIPPED (No backbone)")

    # --- EJECUCIÓN 3: Default Kissat ---
    res, t = run_solver(final_cnf, [], timeout=args.timeout)
    results.append(f"3. Default Kissat         : {res:15s} ({t:.4f}s)")

    # --- EJECUCIÓN 4: Random Phase ---
    res, t = run_solver(final_cnf, [
        "--random_phase_initial",
        "--stable=0"
    ], timeout=args.timeout)
    results.append(f"4. Random Phase           : {res:15s} ({t:.4f}s)")

    print("\nResultados:")
    for r in results:
        print(r)
    print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
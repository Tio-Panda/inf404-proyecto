import argparse
import subprocess
import shutil
import time
import os
import csv
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
SOLVER_PATH = Path("./solver/build/kissat")
CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")
RESULTS_CSV = Path("benchmark_results.csv")
TIMEOUT_DEFAULT = 300 # 5 minutos por problema, ajustable
CFD_DEFAULT = 0.9
# ==============================================================================

def decompress_if_needed(file_path: Path, work_dir: Path) -> Path:
    """
    Si el archivo es .xz, lo descomprime en work_dir.
    Si es texto (.csv, .backbone), lo copia o devuelve tal cual.
    """
    if not file_path.exists():
        return None

    # Caso .xz
    if file_path.suffix == ".xz":
        target_name = file_path.stem # quita .xz
        out_path = work_dir / target_name
        
        if out_path.exists(): return out_path

        # Copia temporal y descompresión
        temp_xz = work_dir / file_path.name
        shutil.copy(file_path, temp_xz)
        try:
            subprocess.run(["xz", "-dkf", str(temp_xz)], check=True)
        except Exception:
            return None # Fallo descompresión
        
        if temp_xz.exists(): os.remove(temp_xz)
        return out_path

    return file_path

def run_solver(cnf_file: Path, extra_args: list[str], timeout: int) -> tuple[str, float]:
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
            if r.returncode == 0: res_str = "UNKNOWN"
            elif err: res_str = "ERROR" # Podrías guardar el error específico si quieres
            else: res_str = f"ERROR (Code {r.returncode})"
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout # Retornamos el tiempo máximo
    except Exception:
        return "PYTHON_ERROR", 0.0
    
    elapsed = time.time() - start
    return res_str, elapsed

def find_backbone(cnf_name: str, backbone_dir: Path) -> Path:
    """
    Busca el backbone correspondiente.
    Asume convención: nombre.cnf -> nombre.cnf.backbone[.xz]
    O: nombre.cnf -> nombre.backbone[.xz]
    """
    # Intentos de nombre
    candidates = [
        cnf_name + ".backbone.xz",
        cnf_name + ".backbone",
        Path(cnf_name).stem + ".backbone.xz", # Si cnf es a.cnf, busca a.backbone.xz
        Path(cnf_name).stem + ".backbone"
    ]
    
    for cand in candidates:
        p = backbone_dir / cand
        if p.exists(): return p
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT)
    p.add_argument("--limit", type=int, default=0, help="Límite de problemas (0 = todos)")
    args = p.parse_args()

    if not SOLVER_PATH.exists():
        print(" Falta el solver (haz make)"); return

    # Preparar CSV
    file_exists = RESULTS_CSV.exists()
    csv_file = open(RESULTS_CSV, 'a', newline='')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["cnf_name", "config", "result", "time", "backbone_found"])

    # Buscar CNFs
    cnfs = sorted(list(CNF_DIR.glob("*.cnf")) + list(CNF_DIR.glob("*.cnf.xz")))
    if args.limit > 0: cnfs = cnfs[:args.limit]

    print(f"Iniciando Benchmark Masivo: {len(cnfs)} problemas")
    print(f" Resultados en: {RESULTS_CSV}")
    
    work_dir = Path("./temp_bench_massive")
    work_dir.mkdir(exist_ok=True)

    for i, raw_cnf in enumerate(cnfs):
        print(f"\n[{i+1}/{len(cnfs)}] Procesando: {raw_cnf.name}")
        
        # Preparar archivos
        cnf_file = decompress_if_needed(raw_cnf, work_dir)
        if not cnf_file: 
            print(" Error descomprimiendo CNF"); continue

        raw_bb = find_backbone(raw_cnf.name, BACKBONE_DIR)
        backbone_file = None
        if raw_bb:
            backbone_file = decompress_if_needed(raw_bb, work_dir)
        
        has_bb = backbone_file is not None
        bb_arg = str(backbone_file.resolve()) if has_bb else ""
        
        configs = [
            ("Default", []),
             # A veces stable=0 ayuda a random
            ("NB-Initial", ["--neural_backbone_initial", f"--backbonefile={bb_arg}", f"--neuroback_cfd={CFD_DEFAULT}"]),
            ("NB-Always", ["--neural_backbone_always", f"--backbonefile={bb_arg}"]),
            ("NB-Target", ["--neural_backbone_target", "--stable=2", f"--backbonefile={bb_arg}", f"--neuroback_cfd={CFD_DEFAULT}"]), # Tu nueva estrella
            ("NB-Modified", ["--neural_backbone_initial", "--neural_backbone_modified", f"--backbonefile={bb_arg}", f"--neuroback_cfd={CFD_DEFAULT}"]) # Tu rebeldía
        ]

        for conf_name, conf_args in configs:
            # Si requiere backbone y no lo tiene, saltar (o marcar como SKIPPED en CSV)
            if "backbonefile" in str(conf_args) and not has_bb:
                print(f"   - {conf_name:12s}: SKIPPED (No backbone)")
                writer.writerow([raw_cnf.name, conf_name, "SKIPPED", 0.0, False])
                continue

            res, t = run_solver(cnf_file, conf_args, args.timeout)
            print(f"   - {conf_name:12s}: {res:15s} ({t:.2f}s)")
            
            # Guardar en CSV
            writer.writerow([raw_cnf.name, conf_name, res, t, has_bb])
            csv_file.flush() # Guardar inmediatamente por si crashea

        # Limpieza por iteración para no llenar disco
        # shutil.rmtree(work_dir) # Descomentar si tienes poco espacio
        # work_dir.mkdir(exist_ok=True)

    csv_file.close()
    print("\nBenchmark finalizado.")

if __name__ == "__main__":
    main()
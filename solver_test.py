import argparse
import subprocess
import shutil
import time
import os
import csv
import re
from pathlib import Path

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
SOLVER_PATH = Path("./solver/build/kissat")
CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")
RESULTS_CSV = Path("benchmark_results.csv")
TIMEOUT_DEFAULT = 120 
CFD_DEFAULT = 0.9
# ==============================================================================

def decompress_if_needed(file_path: Path, work_dir: Path) -> Path:
    if not file_path.exists(): return None
    
    # Caso .xz
    if file_path.suffix == ".xz":
        target_name = file_path.stem 
        out_path = work_dir / target_name
        if out_path.exists(): return out_path # Ya existe descomprimido
        
        temp_xz = work_dir / file_path.name
        shutil.copy(file_path, temp_xz)
        try:
            subprocess.run(["xz", "-dkf", str(temp_xz)], check=True)
        except Exception: return None
        if temp_xz.exists(): os.remove(temp_xz)
        return out_path
    
    # Caso texto plano (ya descomprimido)
    return file_path

def parse_stats(output: str):
    stats = {"conflicts": 0, "decisions": 0, "propagations": 0, "restarts": 0}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("c "): continue
        m = re.match(r'^c\s+([a-zA-Z0-9_]+):\s+(\d+)', line)
        if m:
            name = m.group(1)
            val = int(m.group(2))
            if name in stats: stats[name] = val
    return stats

def run_solver(cnf_file: Path, extra_args: list[str], timeout: int) -> tuple:
    cmd = [str(SOLVER_PATH.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    start = time.time()
    try:
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        out = r.stdout.strip() if r.stdout else ""
        err = r.stderr.strip() if r.stderr else ""
        
        if "SATISFIABLE" in out and "UNSATISFIABLE" not in out: res_str = "SATISFIABLE"
        elif "UNSATISFIABLE" in out: res_str = "UNSATISFIABLE"
        else: res_str = "ERROR" if err else f"UNKNOWN ({r.returncode})"
        
        stats = parse_stats(out)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, {"conflicts": -1}
    except Exception:
        return "PYTHON_ERROR", 0.0, {}
    
    return res_str, time.time() - start, stats

def find_backbone(cnf_stem: str, backbone_dir: Path) -> Path:
    """Busca backbone inteligente para un nombre base de CNF"""
    # Prioridad: Texto plano > Comprimido
    candidates = [
        f"{cnf_stem}.backbone",
        f"{cnf_stem}.cnf.backbone",
        f"{cnf_stem}.backbone.xz",
        f"{cnf_stem}.cnf.backbone.xz"
    ]
    for cand in candidates:
        p = backbone_dir / cand
        if p.exists(): return p
    return None

def get_unique_cnfs(directory: Path):
    """
    Retorna una lista de archivos CNF únicos.
    Si existen 'a.cnf' y 'a.cnf.xz', devuelve solo 'a.cnf'.
    """
    files = list(directory.glob("*.cnf")) + list(directory.glob("*.cnf.xz"))
    unique_problems = {}
    
    for f in files:
        # El nombre base es el identificador único del problema
        # Si es .xz, le quitamos la extensión para obtener el nombre base real
        base_name = f.stem if f.suffix == '.xz' else f.name
        
        # Si ya tenemos una versión de este problema...
        if base_name in unique_problems:
            # ...y la nueva versión es descomprimida (.cnf), reemplazamos la comprimida
            if f.suffix == '.cnf':
                unique_problems[base_name] = f
        else:
            unique_problems[base_name] = f
            
    # Ordenar por nombre para consistencia
    return sorted(unique_problems.values(), key=lambda p: p.name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if not SOLVER_PATH.exists(): print("❌ Falta el solver"); return

    # CSV Header
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(["cnf_name", "config", "result", "time", "conflicts", "decisions", "propagations"])

    # Obtener lista deduplicada
    cnfs = get_unique_cnfs(CNF_DIR)
    if args.limit > 0: cnfs = cnfs[:args.limit]

    print(f"🚀 Benchmarking {len(cnfs)} problemas únicos...")
    work_dir = Path("./temp_bench_massive")
    work_dir.mkdir(exist_ok=True)

    csv_file = open(RESULTS_CSV, 'a', newline='')
    writer = csv.writer(csv_file)

    for i, raw_cnf in enumerate(cnfs):
        print(f"[{i+1}/{len(cnfs)}] {raw_cnf.name} ...")
        
        # Preparar CNF
        cnf_file = decompress_if_needed(raw_cnf, work_dir)
        if not cnf_file: continue

        # Identificar nombre base para buscar backbone (sin .cnf ni .xz)
        # Ej: "problem.cnf" -> "problem" | "problem.cnf.xz" -> "problem"
        problem_stem = raw_cnf.stem if raw_cnf.suffix == '.xz' else raw_cnf.stem
        # Ojo: si el archivo es 'a.cnf', stem es 'a'. Si es 'a.cnf.xz', stem es 'a.cnf'. 
        # Ajuste fino:
        if problem_stem.endswith('.cnf'): problem_stem = problem_stem[:-4]

        raw_bb = find_backbone(problem_stem, BACKBONE_DIR)
        backbone_file = decompress_if_needed(raw_bb, work_dir) if raw_bb else None
        
        has_bb = backbone_file is not None
        bb_arg = str(backbone_file.resolve()) if has_bb else ""
        
        configs = [
            ("Default", ["--stable=2"]),
            ("NB-Initial", ["--neural_backbone_initial", f"--backbonefile={bb_arg}", f"--neuroback_cfd={CFD_DEFAULT}"]),
            ("NB-Always", ["--neural_backbone_always", f"--backbonefile={bb_arg}"]),
            ("NB-Target", ["--neural_backbone_target", "--stable=2", f"--backbonefile={bb_arg}"]),
            ("NB-Modified", ["--neural_backbone_initial", "--neural_backbone_modified", f"--backbonefile={bb_arg}", f"--neuroback_cfd={CFD_DEFAULT}"]),
        ]

        for conf_name, conf_args in configs:
            if "backbonefile" in str(conf_args) and not has_bb:
                writer.writerow([raw_cnf.name, conf_name, "SKIPPED", 0.0, 0, 0, 0])
                continue

            res, t, stats = run_solver(cnf_file, conf_args, args.timeout)
            writer.writerow([
                raw_cnf.name, conf_name, res, f"{t:.4f}", 
                stats.get("conflicts", 0), stats.get("decisions", 0), stats.get("propagations", 0)
            ])
            csv_file.flush()

    csv_file.close()
    print("\n✅ Benchmark finalizado.")

if __name__ == "__main__":
    main()
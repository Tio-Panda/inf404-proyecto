import subprocess
from pathlib import Path
import shutil
import time
import re
import csv

CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")
SOLVER_BINARY = Path("./solver/build/kissat")
RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(exist_ok=True)
COMP_EXTS = [".xz", ".bz2", ".gz", ".lzma"]

# Porcentaje de instancias a tomar (0 < TAKE_FRACTION <= 1.0)
TAKE_FRACTION: float = 1.


# -----------------------------
# Variable global para el modelo
# falta implementar como elegir los backbones dependiendo del modelo

Modelo = "NeuroBack" 
# Modelo = "Mamba"
# -----------------------------

def is_cnf_like(name: str) -> bool:
    lname = name.lower()
    return (".cnf" in lname) or ("_cnf" in lname) or ("dimacs" in lname)

def decompress_file(src_file: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    dest_dir.mkdir(exist_ok=True)
    # remove only final compression suffix
    out_name = src_file.name
    for ext in COMP_EXTS:
        if out_name.endswith(ext):
            out_name = out_name[: -len(ext)]
            break
    decompressed_file = dest_dir / out_name
    temp_src = dest_dir / src_file.name
    if overwrite and decompressed_file.exists():
        decompressed_file.unlink()
    if not decompressed_file.exists():
        shutil.copy(src_file, temp_src)
        print(f"Decompressing {src_file.name} into {decompressed_file}...")
        # choose decompressor based on extension; default to xz-compatible for .xz
        if src_file.suffix == ".xz":
            subprocess.run(["xz", "-dkf", str(temp_src)], check=True)
        elif src_file.suffix == ".gz":
            subprocess.run(["gunzip", "-kf", str(temp_src)], check=True)
        elif src_file.suffix == ".bz2":
            subprocess.run(["bunzip2", "-kf", str(temp_src)], check=True)
        elif src_file.suffix == ".lzma":
            subprocess.run(["unlzma", "-kf", str(temp_src)], check=True)
        else:
            # not compressed: just copy
            shutil.copy(temp_src, decompressed_file)
        # if decompressor produced file without removing ext, ensure destination exists
        if not decompressed_file.exists():
            maybe = dest_dir / src_file.with_suffix("").name
            if maybe.exists():
                decompressed_file = maybe
    return decompressed_file

def run_solver(cnf_file: Path, extra_args: list[str]) -> tuple[str, float]:
    
    cmd = [str(SOLVER_BINARY.resolve()), str(cnf_file.resolve()), "-s", "-n"] + extra_args
    
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=15  # timeout x instancia
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start_time
    elapsed = time.time() - start_time
    if result.stderr:
        print("stderr:", result.stderr.strip())
    stdout = result.stdout.strip() if result.stdout else "NO_OUTPUT"
    return stdout, elapsed


def parse_stats_from_stdout(s: str):
    """Extract main statistics counters from solver stdout.

    Returns dict with keys: conflicts, decisions, propagations, restarts (ints)
    """
    stats = {"conflicts": 0, "decisions": 0, "propagations": 0, "restarts": 0}
    for line in s.splitlines():
        line = line.strip()
        if not line.startswith('c '):
            continue
        m = re.match(r'^c\s+([a-zA-Z0-9_]+):\s+(\d+)', line)
        if m:
            name = m.group(1)
            val = int(m.group(2))
            if name in stats:
                stats[name] = val
    return stats


def export_results_to_csv(summary: dict, output_file: Path = Path(f"Resultados/{Modelo}.csv")):
    """Export summary results to CSV format for cactus plots and comparisons.
    
    Creates a CSV with columns:
    - method: solver configuration name
    - result: SAT/UNSAT/TIMEOUT/ERROR
    - time: solving time in seconds
    - conflicts: number of conflicts
    - decisions: number of decisions
    - propagations: number of propagations
    - restarts: number of restarts
    
    Args:
        summary: Dictionary with method names as keys and list of results as values
        output_file: Path to the output CSV file
    """
    
    rows = []
    
    for method, results in summary.items():
        n=1
        for entry in results:
            _, result_text, solve_time, stats = entry
            
            # Determine result status
            if result_text == "NO_BACKBONE":
                status = "NO_BACKBONE"
            elif result_text == "TIMEOUT":
                status = "TIMEOUT"
            elif result_text == "NO_OUTPUT":
                status = "ERROR"
            elif "UNSATISFIABLE" in result_text:
                status = "UNSAT"
            elif "SATISFIABLE" in result_text:
                status = "SAT"
            else:
                status = "UNKNOWN"
            
            row = {
                "n":n,
                "method": method,
                "result": status,
                "time": f"{solve_time:.4f}",
                "conflicts": stats.get("conflicts", 0),
                "decisions": stats.get("decisions", 0),
                "propagations": stats.get("propagations", 0),
                "restarts": stats.get("restarts", 0),
            }
            rows.append(row)
            n+=1
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ["n", "method", "result", "time", "conflicts", 
                      "decisions", "propagations", "restarts"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"\nResults exported to {output_file}")


def main():
    # Gather CNF-like files (compressed allowed)
    candidate_files = [p for p in CNF_DIR.iterdir() if p.is_file() and is_cnf_like(p.name)]
    cnf_files = sorted(candidate_files, key=lambda p: p.name)

    if 0 < TAKE_FRACTION < 1.0 and cnf_files:
        limit = max(1, int(len(cnf_files) * TAKE_FRACTION))
        cnf_files = cnf_files[:limit]
        print(f"Using a subset: {limit}/{len(candidate_files)} instances.")

    summary = {
        "NeuroBack-Initial": [],
        "NeuroBack-Always": [],
        "NeuroBack-Partial": [],
        "NeuroBack-Prioritized": [],
        "NeuroBack-LowScores": [],
        "Default": [],
    }

    for cnf_file in cnf_files:
        print(f"\n=== Processing {cnf_file.name} ===")
        # Ensure CNF is decompressed to a working path if compressed
        cnf_result_dir = RESULTS_DIR / cnf_file.stem
        cnf_result_dir.mkdir(exist_ok=True)
        cnf_path_for_solver = cnf_file
        if cnf_file.suffix in COMP_EXTS:
            cnf_path_for_solver = decompress_file(cnf_file, cnf_result_dir)

        # Resolve backbone path robustly: accept .backbone or .backbone.xz
        backbone_file = None
        candidates = [
            BACKBONE_DIR / f"{cnf_file.stem}.backbone",
            BACKBONE_DIR / f"{cnf_file.stem}.backbone.xz",
            BACKBONE_DIR / f"{cnf_file.stem}.res",
            BACKBONE_DIR / f"{cnf_file.stem}.res.xz",
        ]
        for bb in candidates:
            if bb.exists():
                cnf_result_dir = RESULTS_DIR / cnf_file.stem
                cnf_result_dir.mkdir(exist_ok=True)
                # if compressed, decompress; else copy to results folder for stable path
                if bb.suffix in COMP_EXTS:
                    backbone_file = decompress_file(bb, cnf_result_dir)
                else:
                    # ensure we have a local copy to pass to solver
                    target = cnf_result_dir / bb.name
                    if not target.exists():
                        shutil.copy(bb, target)
                    backbone_file = target
                break
        if backbone_file is None:
            print(f"No backbone found for {cnf_file.stem} in {BACKBONE_DIR}")

        # NeuroBack-Always
        print("Running NeuroBack-Always...")
        if backbone_file and backbone_file.exists():
            na_out, na_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_always",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            na_stats = parse_stats_from_stdout(na_out)
            summary["NeuroBack-Always"].append((cnf_file.name, na_out, na_time, na_stats))
        else:
            summary["NeuroBack-Always"].append((cnf_file.name, "NO_BACKBONE", 0.0, {"conflicts":0,"decisions":0,"propagations":0,"restarts":0}))

        # NeuroBack-Initial
        print("Running NeuroBack-Initial...")
        if backbone_file and backbone_file.exists():
            nb_out, nb_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_initial",
                "--neuroback_cfd=0.9",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            nb_stats = parse_stats_from_stdout(nb_out)
            summary["NeuroBack-Initial"].append((cnf_file.name, nb_out, nb_time, nb_stats))
        else:
            summary["NeuroBack-Initial"].append((cnf_file.name, "NO_BACKBONE", 0.0, {"conflicts":0,"decisions":0,"propagations":0,"restarts":0}))

        # Default-Kissat
        print("Running Default-Kissat...")
        def_out, def_time = run_solver(cnf_path_for_solver, [
            "--stable=2"
        ])
        def_stats = parse_stats_from_stdout(def_out)
        summary["Default"].append((cnf_file.name, def_out, def_time, def_stats))

        # Partial Neuroback
        print("Running Partial Neuroback...")
        if backbone_file and backbone_file.exists():
            bw_out, bw_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_partial",
                f"--neural_backbone_partial_weight=60",
                f"--backbonefile={backbone_file.resolve()}"
            ])
            bw_stats = parse_stats_from_stdout(bw_out)
            summary["NeuroBack-Partial"].append((cnf_file.name, bw_out, bw_time, bw_stats))
        else:
            summary["NeuroBack-Partial"].append((cnf_file.name, "NO_BACKBONE", 0.0, {"conflicts":0,"decisions":0,"propagations":0,"restarts":0}))

        # Prioritized Neuroback
        print("Running Prioritized Neuroback...")
        if backbone_file and backbone_file.exists():
            pr_out, pr_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_prioritize",
                f"--backbonefile={backbone_file.resolve()}"
            ])
            pr_stats = parse_stats_from_stdout(pr_out)
            summary["NeuroBack-Prioritized"].append((cnf_file.name, pr_out, pr_time, pr_stats))
        else:
            summary["NeuroBack-Prioritized"].append((cnf_file.name, "NO_BACKBONE", 0.0, {"conflicts":0,"decisions":0,"propagations":0,"restarts":0}))

        # LowScored Neuroback
        print("Running LowScored Neuroback...")
        if backbone_file and backbone_file.exists():
            pr_out, pr_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_lowscores",
            
                f"--backbonefile={backbone_file.resolve()}"
            ])
            ls_stats = parse_stats_from_stdout(pr_out)
            summary["NeuroBack-LowScores"].append((cnf_file.name, pr_out, pr_time, ls_stats))
        else:
            summary["NeuroBack-LowScores"].append((cnf_file.name, "NO_BACKBONE", 0.0, {"conflicts":0,"decisions":0,"propagations":0,"restarts":0}))

    # --- PRINT SUMMARY ---
    for config in [ "NeuroBack-Initial" , "NeuroBack-Always",  "Default", "NeuroBack-Partial", "NeuroBack-LowScores","NeuroBack-Prioritized"]:
        print(f"\n===== RESULTS FOR {config} =====")
        total_time = 0
        sat_count = unsat_count = error_count = no_backbone = 0
        sum_conflicts = sum_decisions = sum_propagations = sum_restarts = 0

        for entry in summary[config]:
            # each entry is (cnf_name, result, time, stats_dict)
            cnf_name, result, t, stats = entry
            total_time += t

            # Result can contain many lines (banner, sections, stats). Look
            # for SAT/UNSAT anywhere in the output rather than only at the
            # beginning. Also respect explicit markers we set (NO_BACKBONE,
            # TIMEOUT, NO_OUTPUT).
            if result == "NO_BACKBONE":
                no_backbone += 1
            elif result in ("TIMEOUT", "NO_OUTPUT"):
                error_count += 1
            else:
                # result is solver stdout; search for SAT/UNSAT tokens
                if isinstance(result, str):
                    if "UNSATISFIABLE" in result:
                        unsat_count += 1
                    elif "SATISFIABLE" in result:
                        sat_count += 1
                    else:
                        # neither found: treat as error/misc
                        # keep as error only if no output was produced
                        pass

            # accumulate stats (stats may be missing keys -> default 0)
            sum_conflicts += int(stats.get("conflicts", 0))
            sum_decisions += int(stats.get("decisions", 0))
            sum_propagations += int(stats.get("propagations", 0))
            sum_restarts += int(stats.get("restarts", 0))

        n = len(summary[config]) if summary[config] else 1

        print(f"Total problems: {len(cnf_files)}")
        print(f"SAT: {sat_count}")
        print(f"UNSAT: {unsat_count}")
        print(f"Errors: {error_count}")
        print(f"No Backbone: {no_backbone}")
        print(f"Total solving time: {total_time:.2f}s")
        avg = (total_time / n) if n else 0.0
        print(f"Average per problem: {avg:.2f}s")
        variance = sum((entry[2] - avg) ** 2 for entry in summary[config]) / n if n else 0.0
        print(f"Time variance: {variance:.4f}s^2")

        # Print average internal stats
        print("\nAverage internal statistics:")
        print(f"  Conflicts: {sum_conflicts / n:.2f}")
        print(f"  Decisions: {sum_decisions / n:.2f}")
        print(f"  Propagations: {sum_propagations / n:.2f}")
        print(f"  Restarts: {sum_restarts / n:.2f}")

    # Export results to CSV
    export_results_to_csv(summary)

if __name__ == "__main__":
    main()
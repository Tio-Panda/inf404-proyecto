import subprocess
from pathlib import Path
import shutil
import time
from typing import Tuple, List

CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")
SOLVER_BINARY = Path("./solver/build/kissat")
RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(exist_ok=True)

COMP_EXTS = [".xz", ".bz2", ".gz", ".lzma"]


TAKE_FRACTION: float = 1.0

def remove_compression_suffix(filename: str) -> str:
    """Strip only the compression extension if present (xz, bz2, gz, lzma)."""
    for ext in COMP_EXTS:
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return filename


def is_cnf_like(name: str) -> bool:
    """Heuristic to select CNF-like instances in the dataset dirs.

    We include files whose names contain any of these markers:
    - '.cnf' (typical DIMACS CNF)
    - '_cnf' (mcc2020 naming)
    - 'dimacs' (some benchmarks use '.dimacs')
    """
    lname = name.lower()
    return (".cnf" in lname) or ("_cnf" in lname) or ("dimacs" in lname)


def decompress_file(src_file: Path, dest_dir: Path, overwrite: bool = False) -> Path:
    """Decompress src_file if it's compressed into dest_dir and return the decompressed path.

    The output filename is the source name with only the final compression suffix removed.
    For example:
      - 'foo.cnf.xz'   -> 'dest_dir/foo.cnf'
      - 'bar.b+e.gz'   -> 'dest_dir/bar.b+e'
      - 'baz.lzma'     -> 'dest_dir/baz'
    If the file is not compressed, it is simply copied into dest_dir and that path is returned.
    """
    dest_dir.mkdir(exist_ok=True)

    name_wo_comp = remove_compression_suffix(src_file.name)
    out_path = dest_dir / name_wo_comp

    if out_path.exists() and not overwrite:
        return out_path

    # Work on a temporary copy in destination directory to avoid modifying the source files
    temp_src = dest_dir / src_file.name
    shutil.copy(src_file, temp_src)

    print(f"Decompressing {src_file.name} into {out_path}...")

    if temp_src.suffix == ".xz" or temp_src.suffix == ".lzma":
        # xz can decompress both .xz and .lzma
        subprocess.run(["xz", "-dkf", str(temp_src)], check=True)
    elif temp_src.suffix == ".bz2":
        subprocess.run(["bzip2", "-dkf", str(temp_src)], check=True)
    elif temp_src.suffix == ".gz":
        subprocess.run(["gzip", "-dkf", str(temp_src)], check=True)
    else:
        # Not a recognized compression, just copy as-is
        shutil.move(str(temp_src), str(out_path))
        return out_path

    return out_path


def run_cmd(cmd: List[str]) -> Tuple[str, float]:
    start = time.time()
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start
    elapsed = time.time() - start
    stdout = result.stdout.strip() if result.stdout else "NO_OUTPUT"
    return stdout, elapsed

# Removed build_cmd_default: inline the default solver invocation like other modes



def main():
    # Collect CNF-like files, including compressed variants
    candidate_files = [p for p in CNF_DIR.iterdir() if p.is_file() and is_cnf_like(p.name)]
    cnf_files = sorted(candidate_files, key=lambda p: p.name)

    # Apply subset selection for faster testing
    if 0 < TAKE_FRACTION < 1.0 and cnf_files:
        limit = max(1, int(len(cnf_files) * TAKE_FRACTION))
        cnf_files = cnf_files[:limit]
        print(f"Using a subset: {limit}/{len(candidate_files)} instances.")
    summary = {
        "NeuroBack-Initial": [],
        "NeuroBack-Partial": [],
        "NeuroBack-LowScores": [],
        "NeuroBack-Always": [],
        "Default": []
    }

    for cnf_file in cnf_files:
        print(f"\n=== Processing {cnf_file.name} ===")

        # Prepare a working directory under results for this instance
        instance_key = remove_compression_suffix(cnf_file.name)  
        cnf_result_dir = RESULTS_DIR / Path(instance_key).stem 
        cnf_result_dir.mkdir(exist_ok=True)

        # Ensure we have an uncompressed CNF path to give the solver
        cnf_path_for_solver = decompress_file(cnf_file, cnf_result_dir)

        # Backbone path
        backbone_xz = BACKBONE_DIR / f"{cnf_file.stem}.backbone.xz"
        backbone_file = None
        if backbone_xz.exists():
            backbone_file = decompress_file(backbone_xz, cnf_result_dir)

        # NeuroBack-Always
        print("Running NeuroBack-Always...")
        if backbone_file:
            na_out, na_time = run_cmd([
                str(SOLVER_BINARY.resolve()), str(cnf_path_for_solver.resolve()), "-q", "-n",
                "--stable=2", "--neural_backbone_always",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-Always"].append((cnf_file.name, na_out, na_time))
            
        else:
            summary["NeuroBack-Always"].append((cnf_file.name, "NO_BACKBONE", 0.0))


        # NeuroBack-Initial 
        print("Running NeuroBack-Initial...")
        if backbone_file:
            nb_out, nb_time = run_cmd([
                str(SOLVER_BINARY.resolve()), str(cnf_path_for_solver.resolve()), "-q", "-n",
                "--stable=2", "--neural_backbone_initial", "--neuroback_cfd=0.9",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-Initial"].append((cnf_file.name, nb_out, nb_time))
            
        else:
            summary["NeuroBack-Initial"].append((cnf_file.name, "NO_BACKBONE", 0.8))


        # NeuroBack-Partial 
        print("Running NeuroBack-Partial...")
        if backbone_file:
            bw_out, bw_time = run_cmd([
                str(SOLVER_BINARY.resolve()), str(cnf_path_for_solver.resolve()), "-q", "-n",
                "--stable=2", "--neural_backbone_partial", "--neural_backbone_partial_weight=0.7",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-Partial"].append((cnf_file.name, bw_out, bw_time))
            
        else:
            summary["NeuroBack-Partial"].append((cnf_file.name, "NO_BACKBONE", 0.0))
        

        # NeuroBack-LowScores
        print("Running NeuroBack-LowScores...")
        if backbone_file:
            ls_out, ls_time = run_cmd([
                str(SOLVER_BINARY.resolve()), str(cnf_path_for_solver.resolve()), "-q", "-n",
                "--stable=2", "--neural_backbone_lowscores", "--lowscores_threshold=0.1",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-LowScores"].append((cnf_file.name, ls_out, ls_time))
            
        else:
            summary["NeuroBack-LowScores"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        # Default-Kissat
        print("Running Default-Kissat...")
        def_out, def_time = run_cmd([
            str(SOLVER_BINARY.resolve()), str(cnf_path_for_solver.resolve()),
            "-q", "-n", "--stable=2"
        ])
        summary["Default"].append((cnf_file.name, def_out, def_time))
        



    # --- PRINT SUMMARY ---
    for config in ["NeuroBack-Always","NeuroBack-Initial", "NeuroBack-Partial", "NeuroBack-LowScores", "Default"]:
        print(f"\n===== RESULTS FOR {config} =====")
        total_time = 0
        sat_count = unsat_count = error_count = no_backbone = 0

        for cnf_name, result, t in summary[config]:
            total_time += t

            if "SATISFIABLE" in result:
                sat_count += 1
            elif "UNSATISFIABLE" in result:
                unsat_count += 1
            elif result == "NO_BACKBONE":
                no_backbone += 1
            elif result in ("TIMEOUT", "NO_OUTPUT"):
                error_count += 1
            else:
                # Clasifica cualquier salida no reconocida como error para cerrar la suma
                error_count += 1

        print(f"\n--- METRICS ({config}) ---")
        print(f"Total problems: {len(cnf_files)}")
        print(f"SAT: {sat_count}")
        print(f"UNSAT: {unsat_count}")
        print(f"Errors: {error_count}")
        print(f"No Backbone: {no_backbone}")
        print(f"Total solving time: {total_time:.4f}s")
        avg = (total_time / len(cnf_files)) if cnf_files else 0.0
        print(f"Average per problem: {avg:.4f}s")
        variance = sum((t - avg) ** 2 for _, _, t in summary[config]) / len(cnf_files) if cnf_files else 0.0
        print(f"Time variance: {variance:.4f}s^2")

if __name__ == "__main__":
    main()
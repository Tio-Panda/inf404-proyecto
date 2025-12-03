import subprocess
from pathlib import Path
import shutil
import time

CNF_DIR = Path("./sym_data/cnf/test")
BACKBONE_DIR = Path("./sym_data/backbone/test")
SOLVER_BINARY = Path("./solver/build/kissat")
RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(exist_ok=True)

COMP_EXTS = [".xz", ".bz2", ".gz", ".lzma"]
TAKE_FRACTION: float = 1.0

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
    cmd = [str(SOLVER_BINARY.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=5  # timeout x instancia
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start_time
    elapsed = time.time() - start_time
    if result.stderr:
        print("stderr:", result.stderr.strip())
    stdout = result.stdout.strip() if result.stdout else "NO_OUTPUT"
    return stdout, elapsed

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

        backbone_xz = BACKBONE_DIR / f"{cnf_file.stem}.backbone.xz"
        backbone_file = None
        if backbone_xz.exists():
            cnf_result_dir = RESULTS_DIR / cnf_file.stem
            cnf_result_dir.mkdir(exist_ok=True)
            backbone_file = decompress_file(backbone_xz, cnf_result_dir)

        # NeuroBack-Always
        print("Running NeuroBack-Always...")
        if backbone_file:
            na_out, na_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_always",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-Always"].append((cnf_file.name, na_out, na_time))
        else:
            summary["NeuroBack-Always"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        # NeuroBack-Initial
        print("Running NeuroBack-Initial...")
        if backbone_file:
            nb_out, nb_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_initial",
                "--neuroback_cfd=0.9",
                f"--backbonefile={backbone_file.resolve()}",
            ])
            summary["NeuroBack-Initial"].append((cnf_file.name, nb_out, nb_time))
        else:
            summary["NeuroBack-Initial"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        # Default-Kissat
        print("Running Default-Kissat...")
        def_out, def_time = run_solver(cnf_path_for_solver, [
            "--stable=2"
        ])
        summary["Default"].append((cnf_file.name, def_out, def_time))

        # Partial Neuroback
        print("Running Partial Neuroback...")
        if backbone_file:
            bw_out, bw_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_partial",
                "--neural_backbone_partial_weight=0.7",
                f"--backbonefile={backbone_file.resolve()}"
            ])
            summary["NeuroBack-Partial"].append((cnf_file.name, bw_out, bw_time))
        else:
            summary["NeuroBack-Partial"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        # Prioritized Neuroback
        print("Running Prioritized Neuroback...")
        if backbone_file:
            pr_out, pr_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_prioritize",
                f"--backbonefile={backbone_file.resolve()}"
            ])
            summary["NeuroBack-Prioritized"].append((cnf_file.name, pr_out, pr_time))
        else:
            summary["NeuroBack-Prioritized"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        print("Running LowScored Neuroback...")
        if backbone_file:
            pr_out, pr_time = run_solver(cnf_path_for_solver, [
                "--stable=2",
                "--neural_backbone_lowscores",
                "--lowscores_threshold=0.1",
                f"--backbonefile={backbone_file.resolve()}"
            ])
            summary["NeuroBack-LowScores"].append((cnf_file.name, pr_out, pr_time))
        else:
            summary["NeuroBack-LowScores"].append((cnf_file.name, "NO_BACKBONE", 0.0))

    # --- PRINT SUMMARY ---
    for config in ["NeuroBack-Prioritized", "NeuroBack-Always", "NeuroBack-Initial", "NeuroBack-Partial", "NeuroBack-LowScores", "Default"]:
        print(f"\n===== RESULTS FOR {config} =====")
        total_time = 0
        sat_count = unsat_count = error_count = no_backbone = 0

        for cnf_name, result, t in summary[config]:
            total_time += t

            if result.startswith("SATISFIABLE"):
                sat_count += 1
            elif result.startswith("UNSATISFIABLE"):
                unsat_count += 1
            elif result == "NO_BACKBONE":
                no_backbone += 1
            elif result in ("TIMEOUT", "NO_OUTPUT"):
                error_count += 1

        print(f"\n--- METRICS ({config}) ---")
        print(f"Total problems: {len(cnf_files)}")
        print(f"SAT: {sat_count}")
        print(f"UNSAT: {unsat_count}")
        print(f"Errors: {error_count}")
        print(f"Total solving time: {total_time:.2f}s")
        avg = (total_time / len(cnf_files)) if cnf_files else 0.0
        print(f"Average per problem: {avg:.2f}s")
        variance = sum((t - avg) ** 2 for _, _, t in summary[config]) / len(cnf_files) if cnf_files else 0.0
        print(f"Time variance: {variance:.4f}s^2")

if __name__ == "__main__":
    main()
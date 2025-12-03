import subprocess
from pathlib import Path
import shutil
import time

CNF_DIR = Path("./sym_data/cnf/test/")
BACKBONE_DIR = Path("./sym_data/backbone/test")
SOLVER_BINARY = Path("./solver/build/kissat")
RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(exist_ok=True)

def decompress_file(xz_file: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(exist_ok=True)
    decompressed_file = dest_dir / xz_file.with_suffix("").name
    temp_xz = dest_dir / xz_file.name
    if not decompressed_file.exists():
        shutil.copy(xz_file, temp_xz)
        print(f"Decompressing {xz_file.name} into {decompressed_file}...")
        subprocess.run(["xz", "-dkf", str(temp_xz)], check=True)
    return decompressed_file


def run_solver(cnf_file: Path, extra_args: list[str]) -> tuple[str, float]:
    """Run solver with optional extra arguments and return (result, elapsed_time)"""

    cmd = [str(SOLVER_BINARY.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=5
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start_time

    elapsed = time.time() - start_time

    if result.stderr:
        print("stderr:", result.stderr.strip())

    stdout = result.stdout.strip() if result.stdout else "NO_OUTPUT"
    return stdout, elapsed


def main():
    cnf_files = sorted(CNF_DIR.glob("*.cnf.xz"))
    summary = {"NeuroBack": [], "Default": [], "Random": []}

    for cnf_file in cnf_files:
        print(f"\n=== Processing {cnf_file.name} ===")

        # NeuroBack
        print("Running NeuroBack...")
        backbone_xz = BACKBONE_DIR / f"{cnf_file.stem}.backbone.xz"
        print(backbone_xz)
        if backbone_xz.exists():
            cnf_result_dir = RESULTS_DIR / cnf_file.stem
            cnf_result_dir.mkdir(exist_ok=True)

            backbone_file = decompress_file(backbone_xz, cnf_result_dir)

            nb_out, nb_time = run_solver(cnf_file, [
                "--stable=2",
                "--neural_backbone_initial",
                "--neuroback_cfd=0.9",
                str(backbone_file.resolve())
            ])
            summary["NeuroBack"].append((cnf_file.name, nb_out, nb_time))
        # else:
        #     summary["NeuroBack"].append((cnf_file.name, "NO_BACKBONE", 0.0))

        # Default-Kissat
        # print("Running Default-Kissat...")
        # def_out, def_time = run_solver(cnf_file, [])
        # summary["Default"].append((cnf_file.name, def_out, def_time))

        # Random-Kissats
        # print("Running Random-Kissat...")
        # rand_out, rand_time = run_solver(cnf_file, [
        #     "--seed=42",
        #     "--random_phase_initial=true",
        #     "--tumble=true",
        #     "--stable=0",
        #     "--time=5"
        # ])
        # summary["Random"].append((cnf_file.name, rand_out, rand_time))

    # --- PRINT SUMMARY ---
    for config in ["NeuroBack", "Default", "Random"]:
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
        print(f"Average per problem: {total_time / len(cnf_files):.2f}s")

if __name__ == "__main__":
    main()

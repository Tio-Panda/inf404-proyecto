import csv
import subprocess
from pathlib import Path
import shutil
import time

# =====================
MODEL_NAME = "NeuroBack"   # Cambia aquí: "NeuroBack", "inf404", etc.
MODEL_TAG = MODEL_NAME.lower()

CNF_DIR = Path("./sym_data/cnf/test/")
#BACKBONE_DIR = Path("./sym_data/backbones/test/")
BACKBONE_DIR = Path("./prediction/cpu/cmb_predictions/")
SOLVER_BINARY = Path("./solver/build/kissat")
RESULTS_DIR = Path("./results")

RESULTS_DIR.mkdir(exist_ok=True)

def decompress_file(xz_file: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(exist_ok=True)
    decompressed_file = dest_dir / xz_file.with_suffix("").name
    temp_xz = dest_dir / xz_file.name
    if not decompressed_file.exists():
        shutil.copy(xz_file, temp_xz)
        subprocess.run(["xz", "-dkf", str(temp_xz)], check=True)
    return decompressed_file

def parse_dimacs_header(cnf_path: Path) -> tuple[int, int]:
    with open(cnf_path, "r") as f:
        for line in f:
            if line.startswith("p cnf"):
                _, _, v, c = line.strip().split()
                return int(v), int(c)
    return -1, -1

def run_solver(cnf_file: Path, extra_args: list[str]) -> tuple[str, float]:
    cmd = [str(SOLVER_BINARY.resolve()), str(cnf_file.resolve()), "-q", "-n"] + extra_args
    start = time.time()
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - start

    elapsed = time.time() - start
    stdout = result.stdout.strip() if result.stdout else "NO_OUTPUT"
    return stdout, elapsed

def compute_summary(rows, title):
    total = len(rows)
    sat = sum(1 for r in rows if "SAT" in r["result"])
    unsat = sum(1 for r in rows if "UNSAT" in r["result"])
    errors = sum(1 for r in rows if r["result"] in ["TIMEOUT", "NO_BACKBONE", "NO_OUTPUT"])
    total_time = sum(r["solving_time_sec"] for r in rows)
    avg_time = total_time / total if total > 0 else 0.0

    print(f"--- AGGREGATE METRICS ({title}) ---")
    print(f"Total problems: {total}")
    print(f"SAT: {sat}")
    print(f"UNSAT: {unsat}")
    print(f"Errors: {errors}")
    print(f"Total solving time: {total_time:.2f}s")
    print(f"Average per problem: {avg_time:.2f}s")
    print()

def main():
    cnf_files = sorted(CNF_DIR.glob("*.cnf.xz"))
    model_rows = []
    default_rows = []

    print(f"\n=== Ejecutando benchmark para modelo: {MODEL_NAME} ===\n")

    for cnf_file in cnf_files:
        print(f"\n→ Instancia: {cnf_file.name}")

        cnf_result_dir = RESULTS_DIR / cnf_file.stem
        cnf_result_dir.mkdir(exist_ok=True)

        cnf_uncompressed = decompress_file(cnf_file, cnf_result_dir)
        n_vars, n_clauses = parse_dimacs_header(cnf_uncompressed)

        # ============  MODEL MODE CON BACKBONE ============
        backbone_candidates = list(BACKBONE_DIR.glob(f"{cnf_file.stem}*.res.tar.gz"))

        if backbone_candidates:
            backbone_tar = backbone_candidates[0]
            print(f"  [{MODEL_NAME}] Descomprimiendo backbone: {backbone_tar.name}", end=" ")

            shutil.unpack_archive(backbone_tar, cnf_result_dir)
            extracted = list(cnf_result_dir.glob("*.res")) + list(cnf_result_dir.glob("*.pred"))

            if extracted:
                backbone_file = extracted[0]
                has_backbone = True
                # Ejecutamos el solver con flags de NeuroBack
                extra_args = [
                    "--stable=2",
                    "--neural_backbone_initial",
                    "--neuroback_cfd=0.9",
                    str(backbone_file.resolve())
                ]
                model_out, model_time = run_solver(cnf_uncompressed, extra_args)
                inference_time = 0.0
                print("OK")
            else:
                model_out = "NO_BACKBONE"
                model_time = 0.0
                inference_time = 0.0
                has_backbone = False
                print("ERROR: no se encontró archivo .res dentro del tar")
        else:
            model_out = "NO_BACKBONE"
            model_time = 0.0
            inference_time = 0.0
            has_backbone = False

        print(f"OK → {model_out} ({model_time:.3f}s)")

        model_rows.append({
            "cnf_name": cnf_file.name,
            "solver": MODEL_NAME,
            "result": model_out,
            "solving_time_sec": model_time,
            "inference_time_sec": inference_time,
            "has_backbone": has_backbone,
            "n_vars": n_vars,
            "n_clauses": n_clauses
        })

        # ============ DEFAULT KISSAT ============
        print("  [Default] Ejecutando...", end=" ")
        def_out, def_time = run_solver(cnf_uncompressed, [])
        print(f"OK → {def_out} ({def_time:.3f}s)")

        default_rows.append({
            "cnf_name": cnf_file.name,
            "solver": "Default",
            "result": def_out,
            "solving_time_sec": def_time,
            "inference_time_sec": "",
            "has_backbone": "",
            "n_vars": n_vars,
            "n_clauses": n_clauses
        })

    # ============ SAVE CSV ============
    with open(f"results_{MODEL_TAG}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_rows[0].keys())
        writer.writeheader()
        writer.writerows(model_rows)

    with open("results_default.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=default_rows[0].keys())
        writer.writeheader()
        writer.writerows(default_rows)

    # ============ FINAL SUMMARY ============
    print("\n\n=== FINAL SUMMARY ===\n")
    compute_summary(model_rows, MODEL_NAME)
    compute_summary(default_rows, "Default")


if __name__ == "__main__":
    main()

# NeuroBack: INF404

#### Datasets

# Dataset Preparation (split_dataset.py)

This script must be executed first. It downloads the dataset from HuggingFace, extracts all `.tar.gz` archives, filters CNF files by size, and splits them into pretrain/validation/test sets. The final dataset is stored using symlinks.

## Steps performed by the script
1. **Download** the full NeuroBack dataset into `./full_data/`.
2. **Decompress** all CNF archives into `./processed_data/`.
3. **Filter** CNF files (≤ 6 KB by default).
4. **Create** a combined list of original and dual CNF/backbone pairs.
5. **Split** into:
   - `pretrain/`
   - `validation/`
   - `test/`
6. **Save** the final structure under `./sym_data/` using symbolic links.
## Output structure
```
sym_data/
├── cnf/
│  ├── pretrain/
│  ├── validation/
│  └── test/
└── backbone/
   ├── pretrain/
   ├── validation/
   └── test/
```

You can delete `./full_data/` if you want to save space after running this script, because all necessary files are symlinked in `./sym_data/`.

# PT Dataset Generation (pt_dataset.py)

After splitting the dataset with `split_dataset.py`, you must run `pt_dataset.py`.  
This script converts each CNF + Backbone pair into **PyTorch Geometric `.pt` graph files**.

## What the script does
1. **Reads CNF and Backbone files** from the symbolic split folders (`./sym_data/cnf/...`).
2. **Parses each pair** and builds a bipartite graph representation (variables–clauses).
3. **Generates PyTorch Geometric `Data` objects** with:
   - node features  
   - edges + edge attributes  
   - variable-to-node mappings  
   - backbone labels (when available)
4. **Saves the processed graphs** as `.pt` files inside:

```
data/pt/
├── pretrain/
├── validation/
└── test/
```

Each CNF may produce **multiple graph components**, saved as:

```
<cnf_name>.c-0.pt
<cnf_name>.c-1.pt
...
```

# Training

...


# Solver Testing (solver_test.py)

To use the solver, please compile the solver using the following commands:
```bash
cd solver
./configure && make
cd ..
```
After successful compilation, the solver binary will be available at `solver/build/kissat`.

After compiling the solver, run the `solver_test.py` script which benchmarks three configurations of the Kissat SAT solver:

1. **NeuroBack mode** – runs Kissat using the neural backbone predictions.  
2. **Default Kissat** – standard solver with no extra options.  
3. **Randomized Kissat** – solver with random initialization and limited time.

## What the script does
- Reads CNF files from:  
  `./sym_data/cnf/test/`
- Reads backbone files from:  
  `./sym_data/backbone/test/`
- Decompresses each `*.backbone.xz` into a per-CNF folder inside `./results/`
- Runs the solver three times per file (NeuroBack / Default / Random)
- Collects:
  - SAT / UNSAT / TIMEOUT / errors  
  - Total and average runtime  
- Prints a final summary for all modes.


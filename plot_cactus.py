#!/usr/bin/env python3
"""
Generate comparison plots from solver results.

Usage:
    python3 plot_cactus.py results_comparison.csv
    
The plot shows solving time (y-axis) for each instance (x-axis) in execution order.
This allows direct comparison of methods on the same instances in the same order.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


# -----------------------------
# Variable global para el modelo

Modelo = "NeuroBack" 
# Modelo = "Mamba"

def generate_cactus_plot(csv_file: Path, output_file: Path = None):
    """Generate a cactus-style comparison plot from CSV results.
    
    Plots cumulative solving time (y-axis) for each instance index (x-axis)
    in the original execution order, method by method.
    
    Args:
        csv_file: Path to the CSV file with results
        output_file: Path to save the plot image (optional)
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Filter out non-solved instances
    solved_df = df[df['result'].isin(['SAT', 'UNSAT'])].copy()
    
    # Convert time to numeric
    solved_df['time'] = pd.to_numeric(solved_df['time'])
    
    # Get unique methods
    methods = solved_df['method'].unique()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for idx, method in enumerate(methods):
        method_df = solved_df[solved_df['method'] == method].copy()
        
        # Maintain original execution order and compute cumulative time
        times = method_df['time'].values
        cum_times = times.cumsum() if hasattr(times, 'cumsum') else pd.Series(times).cumsum().values
        
        # X is the instance index (1..N)
        x = range(1, len(method_df) + 1)
        y = cum_times
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(x, y, label=method, linewidth=2, marker=marker, 
                 markersize=4, markevery=max(1, len(x)//20), color=color)
    
    plt.xlabel('Problems solved', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Solver Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()




def main():
    if len(sys.argv) < 2:
        csv_file = Path(f"Resultados/{Modelo}.csv")
        if not csv_file.exists():
            print(f"Error: {csv_file} not found")
            print(f"Usage: {sys.argv[0]} <results_csv_file>")
            sys.exit(1)
    else:
        csv_file = Path(sys.argv[1])
        if not csv_file.exists():
            print(f"Error: {csv_file} not found")
            sys.exit(1)
    
    print(f"Reading results from {csv_file}...")
    
    # Generate comparison plot 
    output_plot = csv_file.parent / f"Resultado_{Modelo}_cactus_plot.png"
    generate_cactus_plot(csv_file, output_plot)
    


if __name__ == "__main__":
    main()

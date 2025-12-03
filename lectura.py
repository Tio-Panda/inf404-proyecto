#!/usr/bin/env python3

from pathlib import Path
import sys

def inspect_backbone(backbone_file: Path):
    """Inspect and display backbone file format and contents."""
    
    if not backbone_file.exists():
        print(f"Error: File not found: {backbone_file}")
        return
    
    print(f"Inspecting: {backbone_file}")
    print(f"File size: {backbone_file.stat().st_size} bytes")
    print("-" * 60)
    
    with open(backbone_file, 'r', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}\n")
    
    # Analyze first 20 lines
    print("First 20 lines (raw):")
    print("-" * 60)
    for i, line in enumerate(lines[:20]):
        print(f"Line {i+1:3d}: {repr(line)}")
    
    print("\n" + "-" * 60)
    print("Analysis:")
    print("-" * 60)
    
    # Check format
    valid_csv = 0      # Format: eidx,score
    valid_ints = 0     # Format: signed integers
    malformed = 0
    empty_lines = 0
    
    for line in lines:
        line = line.strip()
        
        if not line:
            empty_lines += 1
            continue
        
        # Try CSV format (eidx,score)
        if ',' in line:
            try:
                parts = line.split(',')
                if len(parts) == 2:
                    eidx = int(parts[0])
                    score = float(parts[1])
                    valid_csv += 1
                    continue
            except ValueError:
                pass
        
        # Try signed integer format
        try:
            x = int(line)
            valid_ints += 1
            continue
        except ValueError:
            pass
        
        malformed += 1
    
    print(f"✓ CSV format (eidx,score):    {valid_csv} lines")
    print(f"✓ Signed integer format:      {valid_ints} lines")
    print(f"✗ Malformed lines:             {malformed} lines")
    print(f"○ Empty lines:                 {empty_lines} lines")
    print(f"─────────────────────────────")
    print(f"  Total:                       {len(lines)} lines")
    
    # Recommendations
    print("\nRecommendations:")
    if valid_csv > 0 and valid_ints == 0:
        print("→ File is in CSV format (eidx,score)")
        print("  Use: --neural_backbone_initial --neural_backbone_modified")
    elif valid_ints > 0 and valid_csv == 0:
        print("→ File is in signed integer format")
        print("  Use: --neural_backbone_always or --neural_backbone_rephase")
    elif valid_csv > 0 and valid_ints > 0:
        print("⚠ Mixed formats detected - file may be corrupted")
    elif malformed > 0:
        print(f"⚠ {malformed} malformed lines - file format unclear")
        print("  Check file encoding or line endings")

if __name__ == "__main__":
    backbone_file = Path("./sym_data/backbone/test/3_102_423.cnf.backbone")
    
    if len(sys.argv) > 1:
        backbone_file = Path(sys.argv[1])
    
    inspect_backbone(backbone_file)
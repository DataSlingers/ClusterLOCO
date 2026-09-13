from pathlib import Path
import pandas as pd

results_dir = Path("../results/cluster_loco_grid")
outdir = Path('../paper_figures/data')

def consolidate_results(filename, output_filename, output_dir):
    files = sorted(path for path in results_dir.rglob(filename) if path.name != output_filename)
    if not files:
        raise FileNotFoundError(f"No {filename!r} files were found under {results_dir}.")
    print(f"Found {len(files)} files:")
    for path in files:
        print(f"  {path}")

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df

# Combine all the different results into one csv
results_df = consolidate_results(filename="raw_results.csv", output_filename="GMM_results.csv", output_dir=outdir)
summary_df = consolidate_results(filename="summary_results.csv", output_filename="GMM_summary_results.csv", output_dir=outdir)

# Clean up after 
for filename in ("raw_results.csv", "summary_results.csv"):
    for path in results_dir.rglob(filename):
        path.unlink() 

print(results_df["sweep_mode"].value_counts(dropna=False))
print(summary_df["sweep_mode"].value_counts(dropna=False))
print(summary_df[["sweep_mode", "sweep_value", "N", "variance"]].drop_duplicates().sort_values(["sweep_mode", "sweep_value"]).to_string(index=False))
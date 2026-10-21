"""Generate the synthetic datasets of mmm-param-recovery as CSV files.

bench_models.py builds the models from the CSV files.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PRESETS = ["small_business", "medium_business", "large_business", "growing_business"]

# The default data seed of mmm-param-recovery's run_benchmark.py
SEED = sum(map(ord, "mmm_multidimensional"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmm-param-recovery", type=Path, default=HERE / "mmm-param-recovery")
    parser.add_argument("-o", "--output-dir", type=Path, default=HERE / "data" / "mmm")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.mmm_param_recovery))
    from mmm_data_generator import generate_mmm_dataset, get_preset_config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in PRESETS:
        config = get_preset_config(name)
        config.seed = SEED
        data = generate_mmm_dataset(config)["data"]
        path = args.output_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        print(f"Wrote {path} ({len(data)} rows)")


if __name__ == "__main__":
    main()

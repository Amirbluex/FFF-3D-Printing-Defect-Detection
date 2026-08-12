"""
Shared training entry point for all three detectors (YOLOv8, RT-DETR,
DINO). Dispatches to the appropriate wrapper in src/models/ based on
--model. This keeps one consistent interface across Phases 3-5, per
the project's requirement to maintain identical training conditions.
"""

import argparse
import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
CONFIG_PATH = PROJECT_ROOT / "configs" / "shared_hyperparams.yaml"
DATA_YAML_PATH = PROJECT_ROOT / "data" / "processed" / "data.yaml"

MODEL_OUTPUT_DIRS = {
    "yolov8": PROJECT_ROOT / "results" / "baseline_yolov8",
    "rtdetr": PROJECT_ROOT / "results" / "rtdetr",
    "dino": PROJECT_ROOT / "results" / "dino",
}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_wrapper(model_name, config, output_dir):
    if model_name == "yolov8":
        from models.yolov8_wrapper import YOLOv8Wrapper
        return YOLOv8Wrapper(config, DATA_YAML_PATH, output_dir)
    elif model_name == "rtdetr":
        from models.rtdetr_wrapper import RTDETRWrapper  # Phase 4, not yet written
        return RTDETRWrapper(config, DATA_YAML_PATH, output_dir)
    elif model_name == "dino":
        from models.dino_wrapper import DINOWrapper  # Phase 5, not yet written
        return DINOWrapper(config, DATA_YAML_PATH, output_dir)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["yolov8", "rtdetr", "dino"])
    args = parser.parse_args()

    config = load_config()
    output_dir = MODEL_OUTPUT_DIRS[args.model]
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = get_wrapper(args.model, config, output_dir)

    wrapper.train()

    print("\nBenchmarking trained model...")
    benchmark_results = wrapper.benchmark()

    report = {**wrapper.get_report_metadata(), **benchmark_results}

    report_path = output_dir / "performance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"{args.model.upper()} -- FINAL REPORT")
    print(f"{'='*50}")
    for k, v in report.items():
        if k != "attempt_log":
            print(f"  {k}: {v}")
    print(f"\nFull report saved to: {report_path}")
    print(f"Training curves saved to: {output_dir / 'run'}")


if __name__ == "__main__":
    main()
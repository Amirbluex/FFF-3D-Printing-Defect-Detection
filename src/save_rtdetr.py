import json
import yaml
from pathlib import Path
from models.rtdetr_wrapper import RTDETRWrapper
from ultralytics import RTDETR


def main():
    with open("configs/shared_hyperparams.yaml") as f:
        config = yaml.safe_load(f)

    wrapper = RTDETRWrapper(config, "data/processed/data.yaml", "results/rtdetr")
    wrapper.model = RTDETR("results/rtdetr/run/weights/best.pt")
    wrapper.final_batch_size = 8
    wrapper.attempt_log = [{"batch_size": 8, "status": "success (manually stopped at epoch 61, plateaued since ~epoch 30, best checkpoint ~epoch 53)"}]

    results = wrapper.benchmark()
    report = {**wrapper.get_report_metadata(), **results}
    report["epochs_actually_run"] = 61
    report["stopped_reason"] = "manual stop after sustained plateau (patience threshold not yet formally met, but no meaningful improvement since ~epoch 30)"

    Path("results/rtdetr").mkdir(parents=True, exist_ok=True)
    with open("results/rtdetr/performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
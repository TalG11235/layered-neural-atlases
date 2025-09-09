import neural_atlases as na
import json, argparse
from pathlib import Path
from datetime import datetime

def run_training(cfg_path: str, run_dir: str | None = None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run_dir")
    args = ap.parse_args()

    run_training(args.config, args.run_dir)


class neural_atlases_wrapper:
    def __init__(self):
        pass

    def mask(self, vid_path, class_name = None):
        na.preprocess_mask_rcnn.preprocess(
            vid_path=vid_path, 
            class_name=
                class_name if class_name is not None else "anything"
        )

    def optical_flow(self, vid_path, max_long_edge):
        na.preprocess_optical_flow.preprocess(
            vid_path=vid_path,
            max_long_edge=
                max_long_edge if max_long_edge is not None else 768
            )        

    def train(self, training_config, config_path, run_dir = None):
        """Run one experiment; return its run_dir."""

        # decide where to write
        if run_dir is None:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            run_dir = Path("runs") / f"{timestamp}-{config_path.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # patch paths so everything lives under run_dir
        training_config["results_folder_name"] = str(run_dir / "results")
        training_config["checkpoint_path"]     = str(run_dir / "checkpoint" / "checkpoint")
        training_config["run_dir"]             = str(run_dir)

        (run_dir / "cfg.json").write_text(json.dumps(config_path, indent=2))

        na.train.main(training_config)
        return run_dir
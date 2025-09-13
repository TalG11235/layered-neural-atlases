import argparse, os
import yaml

import torch
import semantic_video_compressor as svc
import neural_atlases as na

def main(config_path: str, device: str | None = None, slurm_index: int | None = None):
    # Setup logging
    logger = svc.logger.define_logger()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = yaml.safe_load(open(config_path))

    jobs = config["jobs"]
    if slurm_index is not None:
        jobs = [jobs[slurm_index]]

    device = (device if device is not None else
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for job in jobs:
        svc.pipeline.run_job(job, device=device, logger=logger)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path", type=str, help="Path to the configuration file.")
    ap.add_argument("--device", type=str, default=None, help="Device to run the job on (e.g., 'cuda', 'cpu').")
    ap.add_argument("--slurm_index", type=int, default=None, help="SLURM array index for job selection.")
    args = ap.parse_args()

    main(args.config_path, args.device, args.slurm_index)
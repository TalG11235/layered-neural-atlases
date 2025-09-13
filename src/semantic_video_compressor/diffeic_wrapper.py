import subprocess
import sys
from semantic_video_compressor import logger
from pathlib import Path

def compress_atlases_with_diffeic(
    run_dir,
    input_dir="atlases/",
    output_dir="compressed_atlases/", 
    ckpt_sd="./weight/v2-1_512-ema-pruned.ckpt",
    ckpt_lc="./weight/lc.ckpt",
    config_path="DiffEIC/configs/model/diffeic.yaml",
    steps=50,
    device="cuda"
):
    """
    Compress atlases using DiffEIC inference.
    
    Args:
        input_dir (str): Directory containing atlas images
        output_dir (str): Directory to save compressed atlases
        ckpt_sd (str): Path to stable diffusion checkpoint
        ckpt_lc (str): Path to learned compression checkpoint
        config_path (str): Path to DiffEIC config file
        steps (int): Number of inference steps
        device (str): Device to run on ('cuda' or 'cpu')
    """
    # Ensure output directory exists
    finalized_output_dir = f"{run_dir} / {output_dir}"
    finalized_input_dir = f"{run_dir} / {input_dir}"
    Path(finalized_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python executable
        "DiffEIC/inference_partition.py",
        "--ckpt_sd", str(ckpt_sd),
        "--ckpt_lc", str(ckpt_lc), 
        "--config", str(config_path),
        "--input", str(finalized_input_dir),
        "--output", str(finalized_output_dir),
        "--steps", str(steps),
        "--device", str(device)
    ]

    logger.info(f"Running DiffEIC compression: {' '.join(cmd)}")

    try:
        # Run the command
        result = subprocess.run(
            cmd,
            check=True,  # Raise exception on non-zero exit
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Return strings instead of bytes
            cwd="."  # Run from project root
        )
        
        logger.info("DiffEIC compression completed successfully")
        if result.stdout:
            logger.info("Output:", result.stdout)

        return result
        
    except subprocess.CalledProcessError as e:
        logger.info(f"DiffEIC compression failed with exit code {e.returncode}")
        logger.info("Error output:", e.stderr)
        raise
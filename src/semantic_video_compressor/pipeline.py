from pathlib import Path
from semantic_video_compressor import *
import os
import torch

def run_job(config, device, logger=None):
    # ------- Step 1: Get a video to run on -----------------------------------------------------
    # This could mean one of two things:
    # 1. Download the video as a folder, separated by frames (From the DAVIS dataset)
    # 2. Download the video as a single file, and separate it into frames
    # Either way, at the end of it, we should have a folder named "data" with frames in it.
    input_config = configuration.get_config_section(config, "input")

    if os.path.isfile(input_config.get("path")):
        io.separate_video_to_frames(
            input_config.get("path"), 
            input_config.get("video_name"), 
            input_config.get("frame_rate")
        )
    # -------------------------------------------------------------------------------------------

    # ------- Step 2: Preprocessing -------------------------------------------------------------
    # Run the two preprocessing scripts:
    # 1. `preprocess_mask_rcnn.py` to segement the video to front and back
    # 2. `preprocess_optical_flow.py` to get the optical flow of the video using RAFT
    na = neural_atlases_wrapper()

    masking_config = configuration.get_config_section(config, "masking")
    na.mask(
        vid_path="data/" + input_config.get("video_name"), 
        class_name=masking_config.get("class_name")
    )

    optical_flow_config = configuration.get_config_section(config, "optical_flow")
    na.optical_flow(
        vid_path="data/" + input_config.get("video_name"), 
        max_long_edge=optical_flow_config.get("max_long_edge")
    )
    # -------------------------------------------------------------------------------------------

    # ------- Step 3: Run the training job ------------------------------------------------------
    # This is as simple as activating the neural_atlases conda environment and running the 
    # training script, which is train.py.
    training_config = configuration.get_config_section(config, "training")
    run_dir = na.train(training_config)
    # -------------------------------------------------------------------------------------------

    # ------- Step 4: Extract relevant data from training result --------------------------------
    # After the training is done, we need to extract the texture files from the results.
    # This is done by finding the latest subdirectory in the results directory, and copying the
    # texture files to the atlases directory.
    training_output_config = configuration.get_config_section(config, "training_output")
    na.extract_atlases_from_results(run_dir + training_output_config.get("results_dir"), 
        run_dir + training_output_config.get("output_dir"))
    # -------------------------------------------------------------------------------------------

    # ------- Step 5: Compress the atlases ------------------------------------------------------
    # Compress the atlases using RDEIC, and move the compressed atlases to the
    # compressed_atlases directory.
    compression_config = configuration.get_config_section(config, "compression")
    diffeic_wrapper.compress_atlases_with_diffeic(run_dir,
        input_dir=compression_config.get("input_dir"),
        output_dir=compression_config.get("output_dir"),
        ckpt_sd=compression_config.get("ckpt_sd"),
        ckpt_lc=compression_config.get("ckpt_lc"),
        config_path=compression_config.get("config_path"),
        steps=compression_config.get("steps"),
        device=device)
    # -------------------------------------------------------------------------------------------

    # ------- Step 6: Save Checkpoint -----------------------------------------------------------
    # Save the checkpoint of the training job to a file named checkpoint.pth in the checkpoint 
    # directory.
    checkpoint.normalize_checkpoint(run_dir)
    # -------------------------------------------------------------------------------------------

    # ------- Step 7: Add alpha channels to atlases ---------------------------------------------
    # Add alpha channels to the atlases using the alpha mattes generated during the masking
    # step, and move the atlases with alpha channels to the compressed_atlases_alpha
    # directory.
    alpha_config = configuration.get_config_section(config, "alpha")
    na.add_alpha_channels_to_atlases(
        rgb_folder=Path(run_dir) / compression_config.get("output_dir"),
        output_folder=Path(run_dir) / alpha_config.get("output_dir"),
        alpha_folder=Path("data") / input_config.get("video_name") / "masks"
    )
    # -------------------------------------------------------------------------------------------

    # ------- Step 8: Quantize ------------------------------------------------------------------
    # Quantize the atlases using the quantization script, and move the quantized atlases to the
    # compressed_editing_outputs directory.
    quantization_config = configuration.get_config_section(config, "quantization")
    checkpoint.quantize_checkpoint(
        quantization_config,
        device=device,
        input_checkpoint_path=Path(run_dir) / "checkpoint",
        output_checkpoint_path=Path(run_dir) / "checkpoint"
    )
    # -------------------------------------------------------------------------------------------

    # ------- Step 9: Reconstruct the video -----------------------------------------------------
    # Reconstruct the video using the editing script, and save the output to the
    # compressed_editing_outputs directory.
    reconstruction_config = configuration.get_config_section(config, "reconstruction")
    na.reconstruct(reconstruction_config, run_dir, device)
    # -------------------------------------------------------------------------------------------

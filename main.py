import os
import semantic_video_compressor as svc
import neural_atlases as na

if __name__ == "__main__":
    # ------- Step 0: Setup -----------------------------------------------------
    # Setup logging
    logger = svc.logger.define_logger()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = svc.configuration.load_config("config/config.json")
    
    # Clear previous data
    svc.io.clear_previous_data()
    # -------------------------------------------------------------------------------------------

    # ------- Step 1: Get a video to run on -----------------------------------------------------
    # This could mean one of two things:
    # 1. Download the video as a folder, separated by frames (From the DAVIS dataset)
    # 2. Download the video as a single file, and separate it into frames
    # Either way, at the end of it, we should have a folder named "data" with frames in it.
    input_config = svc.configuration.get_input_config(config)

    if os.path.isfile(input_config.get("path")):
        svc.io.separate_video_to_frames(
            input_config.get("path"), 
            input_config.get("video_name"), 
            input_config.get("frame_rate")
        )
    # -------------------------------------------------------------------------------------------

    # ------- Step 2: Preprocessing -------------------------------------------------------------
    # Run the two preprocessing scripts:
    # 1. `preprocess_mask_rcnn.py` to segement the video to front and back
    # 2. `preprocess_optical_flow.py` to get the optical flow of the video using RAFT
    masking_config = svc.configuration.get_masking_config(config)

    na.preprocess_mask_rcnn.preprocess(
        vid_path="data/" + input_config.get("video_name"), 
        class_name=masking_config.get("class_name")
    )

    # -------------------------------------------------------------------------------------------

    # ------- Step 3: Run the training job ------------------------------------------------------
    # This is as simple as activating the neural_atlases conda environment and running the 
    # training script, which is train.py.
    # -------------------------------------------------------------------------------------------

    # ------- Step 4: Extract relevant data from training result --------------------------------
    # After the training is done, we need to extract the texture files from the results.
    # This is done by finding the latest subdirectory in the results directory, and copying the
    # texture files to the atlases directory.
    # -------------------------------------------------------------------------------------------

    # ------- Step 5: Compress the atlases ------------------------------------------------------
    # Compress the atlases using RDEIC, and move the compressed atlases to the
    # compressed_atlases directory.
    # -------------------------------------------------------------------------------------------

    # ------- Step 6: Save Checkpoint -----------------------------------------------------------
    # Save the checkpoint of the training job to a file named checkpoint.pth in the checkpoint 
    # directory.
    # -------------------------------------------------------------------------------------------

    # ------- Step 7: Add alpha channels to atlases ---------------------------------------------
    # \
    # -------------------------------------------------------------------------------------------

    # ------- Step 8: Quantize ------------------------------------------------------------------
    # Quantize the atlases using the quantization script, and move the quantized atlases to the
    # compressed_editing_outputs directory.
    # -------------------------------------------------------------------------------------------

    # ------- Step 9: Reconstruct the video -----------------------------------------------------
    # Reconstruct the video using the editing script, and save the output to the
    # compressed_editing_outputs directory.
    # -------------------------------------------------------------------------------------------
    pass
import neural_atlases as na
import json, argparse
import torch
from pathlib import Path
from datetime import datetime
import os
import shutil
import imageio
import numpy as np
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

    def train(self, training_config, run_dir = None, device = None):
        if run_dir is None:
            timestamp = datetime.now(datetime.timezone.local).strftime("%d%m%YT%H%M%S")
            run_dir = Path("runs") / f"{timestamp}-{training_config['video_name']}"
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "config.json").write_text(json.dumps(training_config, indent=2))
        
        na.train.main(training_config, run_dir, device)
        return run_dir
    
    def reconstruct(self, reconstruction_config, run_dir, device = None):
        na.reconstruct.main(reconstruction_config, run_dir, device)
    
    
    def extract_atlases_from_results(results_dir="train_results", output_dir="atlases"):
        """
        Extract texture atlases from training results.
        
        Args:
            results_dir (str): Directory containing training results
            output_dir (str): Directory to save extracted atlases
        """
        # Find the experiment directory (only directory inside results/)
        results_path = Path(results_dir)

        # Find the subdirectory with the highest number (latest frame index)
        evaluation_dirs = [d for d in results_path.iterdir() if d.is_dir()]
        if not evaluation_dirs:
            raise ValueError(f"No subdirectories found in {results_dir}")

        # Sort numerically and take the last one
        latest_subdir = sorted(evaluation_dirs, key=lambda x: int(x.name) if x.name.isdigit() else 0)[-1]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy texture files
        texture_files = ["texture_orig1.png", "texture_orig2.png"]
        copied_files = []
        
        for texture_file in texture_files:
            source_path = latest_subdir / texture_file
            dest_path = Path(output_dir) / texture_file
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                copied_files.append(texture_file)
                print(f"Extracted {texture_file} to {output_dir}")
            else:
                print(f"Warning: {texture_file} not found in {latest_subdir}")

    def add_alpha_channels_to_atlases(
        self,
        rgb_folder="compressed_atlases/",
        output_folder="compressed_atlases_alpha/",
        alpha_folder=None
    ):
        """
        Add alpha channels to compressed atlas images.
        
        Args:
            rgb_folder (str): Directory containing RGB atlas images
            output_folder (str): Directory to save RGBA atlas images
            alpha_folder (str, optional): Directory containing alpha channel images.
                                        If None, adds constant alpha=1
        """
        rgb_path = Path(rgb_folder)
        output_path = Path(output_folder)
        alpha_path = Path(alpha_folder) if alpha_folder else None
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all PNG files in RGB folder
        rgb_files = list(rgb_path.glob("*.png"))
        
        if not rgb_files:
            print(f"No PNG files found in {rgb_folder}")
            return
        
        print(f"Processing {len(rgb_files)} atlas images...")
        
        for rgb_file in rgb_files:
            filename = rgb_file.name
            output_file = output_path / filename
            alpha_file = alpha_path / filename if alpha_path else None
            
            # Check if alpha file exists
            if alpha_file and alpha_file.exists():
                print(f"Adding alpha from {alpha_file} to {filename}")
                self._add_custom_alpha(rgb_file, alpha_file, output_file)
            else:
                print(f"No alpha image found. Adding constant alpha=1 to {filename}")
                self._add_constant_alpha(rgb_file, output_file)

        print(f"✅ Alpha channels added. Results saved to {output_folder}")

    def _add_custom_alpha(self, rgb_file, alpha_file, output_file):
        """Add custom alpha channel from separate alpha image."""
        try:
            # Load RGB and alpha images
            rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
            alpha = imageio.imread(alpha_file).astype(np.float32) / 255.0
            
            # Handle different alpha formats
            if alpha.ndim == 3:
                # Use alpha channel if RGBA, or first channel if RGB
                alpha = alpha[:, :, 3] if alpha.shape[2] == 4 else alpha[:, :, 0]
            
            # Ensure alpha is 2D
            if alpha.ndim != 2:
                raise ValueError(f"Alpha image has unexpected dimensions: {alpha.shape}")
            
            # Combine RGB + Alpha
            rgba = np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)
            
            # Save as RGBA
            imageio.imwrite(output_file, (rgba * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"Error processing {rgb_file} with alpha {alpha_file}: {e}")
            # Fallback to constant alpha
            self._add_constant_alpha(rgb_file, output_file)

    def _add_constant_alpha(self, rgb_file, output_file):
        """Add constant alpha=1 to RGB image."""
        try:
            # Load RGB image
            rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
            
            # Create constant alpha channel
            alpha = np.ones(rgb.shape[:2], dtype=np.float32)
            
            # Combine RGB + Alpha
            rgba = np.concatenate([rgb[:, :, :3], alpha[:, :, None]], axis=-1)
            
            # Save as RGBA
            imageio.imwrite(output_file, (rgba * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"Error processing {rgb_file}: {e}")
            raise

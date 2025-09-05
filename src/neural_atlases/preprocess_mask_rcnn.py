from detectron2.utils.logger import setup_logger
setup_logger()
import argparse
from pathlib import Path
import numpy as np
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.image as mpimg

def create_args_namespace(vid_path, class_name):
    return argparse.Namespace(
        vid_path=Path(vid_path), 
        class_name=class_name
    )

def preprocess(args=None, vid_path=None, class_name=None):
    """
    Preprocess video frames with MaskRCNN.
    
    Can be called in two ways:
    1. CLI mode: preprocess(args) where args is from argparse
    2. Programmatic: preprocess(vid_path='data', class_name='person')
    
    Args:
        args: argparse.Namespace object (for CLI compatibility)
        vid_path: str/Path to video frames directory (for programmatic calls)
        class_name: str, object class to segment (for programmatic calls)
        **kwargs: additional parameters
    """
    # If called programmatically, create args object
    if args is None:
        args = create_args_namespace(vid_path, class_name)

    images = sorted(args.vid_path.glob('*.jpg'))
    vid_name = args.vid_path.name
    vid_root = args.vid_path.parent
    out_mask_dir = vid_root / f'{vid_name}_maskrcnn'
    out_mask_dir.mkdir(exist_ok=True)


    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    number_of_frames = len(images)

    for i in range(0,number_of_frames):
        # try:
        im = np.array(mpimg.imread(images[i]))
        outputs = predictor(im)
        if args.class_name == 'anything':
           try:
                mask = outputs["instances"].pred_masks[0].cpu().numpy()
                cv2.imwrite(f"{out_mask_dir}/%05d.png" % (i), mask * 255.0)
           except:
                cv2.imwrite(f"{out_mask_dir}/%05d.png" % (i), np.zeros((im.shape[0], im.shape[1])))
        else:
            found_anything = False
            for j in range(len(outputs['instances'])):
                if predictor.metadata.thing_classes[(outputs['instances'][j].pred_classes.cpu()).long()]==args.class_name:
                    # found the required class, save the mask
                    mask = outputs["instances"].pred_masks[j].cpu().numpy()
                    cv2.imwrite(f"{out_mask_dir}/%05d.png"%(i),  mask * 255.0)
                    found_anything = True
                    break
                else:
                    # found unneeded class
                    print("Frame %d: Did not find %s, found %s"%(i,args.class_name,predictor.metadata.thing_classes[(outputs['instances'][j].pred_classes.cpu()).long()]))
            if not found_anything:
                cv2.imwrite(f"{out_mask_dir}/%05d.png" % (i), np.zeros((im.shape[0],im.shape[1])))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MaskRCNN preprocessing for video frames')
    parser.add_argument('--vid_path', type=Path, required=True,
                       help='Path to directory containing video frames')
    parser.add_argument('--class_name', type=str, default='person',
                       help='Object class to segment (person, car, anything, etc.)')
    return parser.parse_args()

if __name__ == '__main__':
    """CLI entry point."""
    args = parse_args()
    preprocess(args)
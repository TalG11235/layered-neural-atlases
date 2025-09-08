import neural_atlases as na

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
        
    def train(self, training_config):
        na.train.main(training_config)
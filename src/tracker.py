
from __future__ import print_function
from helper.image_proc import cropPadImage
from helper.BoundingBox import BoundingBox, calculate_box
from helper.config import POLICY
import os
import cv2
import numpy as np

class bbox_estimator:
    """tracker class"""

    def __init__(self, show_intermediate_output, logger):
        """TODO: to be defined. """
        self.show_intermediate_output = show_intermediate_output
        self.logger = logger

    def init(self, image_curr, bbox_gt):
        """ initializing the first frame in the video
        """
        self.image_prev = image_curr
        self.bbox_prev_tight = bbox_gt
        self.bbox_curr_prior_tight = bbox_gt

    def preprocess(self, image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO

        """
        # num_channels = image.shape[-1]
        # if num_channels == 1 and image.shape[2] == 3:
        #     image_out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # elif num_channels == 1 and image.shape[2] == 4:
        #     image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        # elif num_channels == 3 and image.shape[2] == 4:
        #     image_out = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # elif num_channels == 3 and image.shape[2] == 1:
        #     image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # else:
        #     image_out = image

        image_out = image
        if image_out.shape != (POLICY['HEIGHT'], POLICY['WIDTH'], POLICY['channels']):
            image_out = cv2.resize(image_out, (POLICY['WIDTH'], POLICY['HEIGHT']), interpolation=cv2.INTER_LINEAR)

        image_out = np.float32(image_out)
        # image_out -= np.array(self.mean)
        # image_out = np.transpose(image_out, [2, 0, 1])        # caffe [n c h w], tf [n h w c]
        return image_out

    def track(self, image_curr, tracknet, sess):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        target_pad, _, _,  _ = cropPadImage(self.bbox_prev_tight, self.image_prev)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(self.bbox_curr_prior_tight, image_curr)

        # image, BGR(training type)
        cur_search_region_resize = self.preprocess(cur_search_region)
        target_pad_resize = self.preprocess(target_pad)

        cur_search_region_expdim = np.expand_dims(cur_search_region_resize, axis=0)
        target_pad_expdim = np.expand_dims(target_pad_resize, axis=0)

        fc4 = sess.run([tracknet.fc4], feed_dict={tracknet.image: cur_search_region_expdim,
                                                  tracknet.target: target_pad_expdim})
        bbox_estimate = calculate_box(fc4)
        # this box is NMS result, TODO, all bbox check
        if not len(bbox_estimate) == 0:
            bbox_estimate = BoundingBox(bbox_estimate[0][0], bbox_estimate[0][1], bbox_estimate[0][2], bbox_estimate[0][3])

            # Inplace correction of bounding box
            bbox_estimate.unscale(cur_search_region)
            bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)

            self.image_prev = image_curr
            self.bbox_prev_tight = bbox_estimate
            self.bbox_curr_prior_tight = bbox_estimate
        else:
            bbox_estimate = False

        return bbox_estimate
from rastervision.pytorch_learner import(
    ObjectDetectionRandomWindowGeoDataset
)
import numpy as np
from rastervision.core.box import Box
from rastervision.core.data import ObjectDetectionLabels
import random

import logging
log = logging.getLogger(__name__)
class CustomODRandomWindowGeoDataset(ObjectDetectionRandomWindowGeoDataset):

    def _sample_pos_window(self)->Box:
        """Sample a window containing at least one bounding box.

        This is done by randomly sampling one of the bounding boxes in the
        scene and drawing a random window around it.
        """
        bbox: Box = np.random.choice(self.bboxes)
        box_h, box_w = bbox.size

        # check if it is possible to sample a containing window
        hmax, wmax = self.max_size
        if box_h > hmax or box_w > wmax:
            self.scene.label_source.ioa_thresh=.5
            ymin, xmin, ymax, xmax = [int(coord) for coord in bbox.tuple_format()]
            bbox=Box(ymin,xmin,ymax,xmax)
            windows_over_bbox:list = bbox.get_windows(self.max_size,self.max_size)
            return random.choice(windows_over_bbox)

        # try to sample a window size that is larger than the box's size
        self.scene.label_source.ioa_thresh=.9
        for _ in range(self.max_sample_attempts):
            h, w = self.sample_window_size()
            if h >= box_h and w >= box_w:
                window = bbox.make_random_box_container(h, w)
                return window
        log.warning('ObjectDetectionRandomWindowGeoDataset: Failed to find '
                    'suitable (h, w) for positive window. '
                    f'Using (hmax, wmax) = ({hmax}, {wmax}) instead.')
        window = bbox.make_random_box_container(hmax, wmax)
        return window
    
    
    # def sample_window(self):
    #     """Sample a window with random size and location within the AOI.

    #     If the scene has AOI polygons, try to find a random window that is
    #     within the AOI. Otherwise, just return the first sampled window.

    #     Raises:
    #         StopIteration: If unable to find a valid window within
    #             self.max_sample_attempts attempts.

    #     Returns:
    #         Box: The sampled window.
    #     """
    #     if not self.has_aoi_polygons:
    #         window = self._generate_window()
    #         return window

    #     for _ in range(self.max_sample_attempts):
    #         window = self._generate_window()
    #         if self.within_aoi:
    #             if Box.within_aoi(window, self.aoi):
    #                 return window
    #         else:
    #             if Box.intersects_aoi(window, self.aoi):
    #                 return window
    #     log.warning('Failed to find valid window within scene AOI in '
    #                         f'{self.max_sample_attempts} attempts, returning an invalid one.')
    #     return window
        
        



    def _generate_window(self):
        """Randomly sample a window with random size and location."""
        h, w = self.sample_window_size()
        x, y = self.sample_window_loc(h, w)
        window = Box(y, x, y + h, x + w)
        return window
# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/processing/single_file_processor.py

import numpy as np
import logging
from config.reference_paths import target_data_info
from modules.alignment.pointcloud_alignment import PointCloudAlignment  # assumed location

logging.basicConfig(level=logging.INFO)

class SingleFileProcessor:
    """
    Processes a point cloud for a single vertebra (C1 or C2):
    - Aligns to a reference model
    - Normalizes the cloud
    - Samples fixed number of points
    """

    def __init__(self, npoints: int = 6000):
        self.npoints = npoints

    @staticmethod
    def pc_normalize(pc: np.ndarray) -> tuple:
        """
        Normalize the point cloud to zero-mean and scale to unit sphere.
        """
        centroid = np.mean(pc, axis=0)
        pc_centered = pc - centroid
        scale = np.max(np.linalg.norm(pc_centered, axis=1))
        pc_normalized = pc_centered / scale
        return pc_normalized, centroid, scale

    def process(self, point_cloud_np: np.ndarray, category: str) -> tuple:
        """
        Aligns, normalizes, and samples the input point cloud.
        Returns sampled points, centroid, scale factor, and transformation matrix.
        """
        if category not in target_data_info:
            logging.error(f"Invalid category: {category}. Must be 'C1' or 'C2'.")
            return None, None, None, None

        try:
            aligner = PointCloudAlignment(target_data_info[category])
            aligned, transform_matrix, best_match = aligner.register_and_align_by_pca(point_cloud_np)
            logging.info(f"Alignment done. Best match: {best_match}")
        except Exception as e:
            logging.error(f"Alignment failed: {e}")
            return None, None, None, None

        normalized, centroid, scale = self.pc_normalize(aligned)

        # Sample npoints
        total_points = normalized.shape[0]
        indices = np.random.choice(total_points, self.npoints, replace=total_points < self.npoints)
        sampled_points = normalized[indices]

        return sampled_points, centroid, scale, transform_matrix

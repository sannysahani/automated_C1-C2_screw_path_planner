# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/alignment/pointcloud_alignment.py

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)

class PointCloudAlignment:
    """
    Aligns input point clouds to the best-matching reference using PCA and Open3D registration.
    """

    def __init__(self, target_data_info: dict):
        """
        Initialize with dictionary of reference point cloud paths.
        """
        self.target_data_info = target_data_info
        self.targets = {}
        self.load_target_data()

    def load_target_data(self):
        """Load and store reference point clouds from disk."""
        for category, file_path in self.target_data_info.items():
            try:
                self.targets[category] = self._load_point_cloud(file_path)
                logging.info(f"Loaded target point cloud for: {category}")
            except Exception as e:
                logging.error(f"Failed loading {category} from {file_path}: {e}")

    @staticmethod
    def _load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
        """Load .txt file into Open3D point cloud."""
        data = np.loadtxt(file_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        return pcd

    @staticmethod
    def _numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
        """Convert NumPy array to Open3D point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    @staticmethod
    def _compute_pca_eigenvalues(points: np.ndarray) -> np.ndarray:
        """Return sorted PCA eigenvalues (descending)."""
        pca = PCA(n_components=3)
        pca.fit(points)
        return np.sort(pca.explained_variance_)[::-1]

    def find_best_match_by_size(self, source_points: np.ndarray) -> tuple:
        """
        Find target point cloud that best matches source based on PCA size.
        """
        source_eigen = self._compute_pca_eigenvalues(source_points)
        min_diff = float('inf')
        best_target = None
        best_category = None

        for category, pcd in self.targets.items():
            target_eigen = self._compute_pca_eigenvalues(np.asarray(pcd.points))
            diff = np.linalg.norm(source_eigen - target_eigen)
            if diff < min_diff:
                min_diff = diff
                best_target = pcd
                best_category = category

        return best_target, best_category

    @staticmethod
    def _compute_fpfh_features(pcd, voxel_size):
        """Compute normals and FPFH features."""
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return pcd, fpfh

    def _register_point_clouds(self, source, target, voxel_size=4):
        """Align source to target using RANSAC + ICP."""
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        source_down, source_fpfh = self._compute_fpfh_features(source_down, voxel_size)
        target_down, target_fpfh = self._compute_fpfh_features(target_down, voxel_size)

        dist_thresh = voxel_size * 7
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 3000)
        )

        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 4, ransac_result.transformation)

        return result_icp

    def register_and_align_by_pca(self, points: np.ndarray) -> tuple:
        """
        Align source points with best-matching target.

        Returns:
            - Aligned points (np.ndarray)
            - Transformation matrix (np.ndarray)
            - Best category name (str)
        """
        source_pcd = self._numpy_to_pcd(points)
        target_pcd, category = self.find_best_match_by_size(points)

        try:
            result = self._register_point_clouds(source_pcd, target_pcd)
            aligned = source_pcd.transform(result.transformation)
            logging.info(f"Aligned with target category: {category}")
            return np.asarray(aligned.points), result.transformation, category
        except Exception as e:
            logging.error(f"Registration failed: {e}")
            return points, np.identity(4), None

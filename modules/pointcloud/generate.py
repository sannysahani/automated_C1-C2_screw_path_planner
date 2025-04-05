# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/pointcloud/generate.py

import open3d as o3d
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class PointCloudGenerator:
    """Utility class to generate a point cloud from an STL/OBJ file."""

    @staticmethod
    def process_model_to_point_cloud(file_path: str, number_of_points: int = 5000) -> o3d.geometry.PointCloud:
        """Convert a 3D mesh file (.stl or .obj) to a point cloud with normals."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()
            mesh.normalize_normals()
            point_cloud = mesh.sample_points_uniformly(number_of_points=number_of_points)
            logging.info(f"Generated point cloud with {len(point_cloud.points)} points.")
            return point_cloud
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            raise

    @staticmethod
    def save_point_cloud_with_normals(point_cloud: o3d.geometry.PointCloud, output_file: str):
        """Save point cloud (with normals) to a .txt file."""
        try:
            points = np.asarray(point_cloud.points)
            normals = np.asarray(point_cloud.normals)
            data = np.hstack((points, normals))
            np.savetxt(output_file, data, fmt="%.6f")
            logging.info(f"Point cloud saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving to {output_file}: {e}")
            raise

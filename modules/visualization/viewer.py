# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/visualization/viewer.py

import open3d as o3d
import numpy as np
import os

def load_stl_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # light gray for bones
    return mesh

def load_screw_path(file_path: str, color=[1, 0, 0]) -> o3d.geometry.PointCloud:
    try:
        points = np.loadtxt(file_path)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd
    except Exception as e:
        print(f"❌ Could not load screw path {file_path}: {e}")
        return None

def launch_viewer(model_files: list, screw_paths_root: str):
    geometries = []

    # Load STL meshes
    for mesh_path in model_files:
        if mesh_path.lower().endswith(".stl") or mesh_path.lower().endswith(".obj"):
            mesh = load_stl_mesh(mesh_path)
            geometries.append(mesh)

    # Define a color palette
    color_palette = [
        [1, 0, 0],     # Red
        [0, 1, 0],     # Green
        [0, 0, 1],     # Blue
        [1, 0.5, 0],   # Orange
        [0.5, 0, 1],   # Purple
        [0, 1, 1],     # Cyan
        [1, 0, 1],     # Magenta
        [0.7, 0.7, 0], # Olive
        [0, 0.7, 0.7], # Teal
    ]

    pair_color_map = {}
    color_idx = 0

    # Load screw paths recursively
    for root, _, files in os.walk(screw_paths_root):
        for file in files:
            if not file.endswith(".txt"):
                continue

            pair_name = "_".join(file.split("_")[:2])  # Example: CrEntry_CrExit from CrEntry_CrExit_screw_1.txt

            # Assign same color to same pair
            if pair_name not in pair_color_map:
                pair_color_map[pair_name] = color_palette[color_idx % len(color_palette)]
                color_idx += 1

            color = pair_color_map[pair_name]
            screw_path = os.path.join(root, file)
            pcd = load_screw_path(screw_path, color=color)
            if pcd:
                geometries.append(pcd)

    if geometries:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="🦴 Screw Path Viewer",
            width=1400,
            height=900,
            mesh_show_back_face=True
        )
    else:
        print("⚠️ No geometries loaded for visualization.")


# Example usage:
if __name__ == "__main__":
    # Load all STL files from input_models
    model_dir = "data/input_models"
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".stl")]

    # Path to screw path output (with C1/C2 subfolders)
    screw_path_dir = "data/screw_paths"

    launch_viewer(model_files, screw_path_dir)

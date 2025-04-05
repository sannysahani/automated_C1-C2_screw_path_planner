# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/path_generation/screw_path_planner.py

import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class ScrewPathPlanner:
    def __init__(self, total_length: float = 52, radius: float = 1.7, num_points: int = 100):
        self.total_length = total_length
        self.radius = radius
        self.num_points = num_points

    @staticmethod
    def adjust_screw_to_length(entry: np.ndarray, exit: np.ndarray, length: float) -> tuple:
        vec = exit - entry
        current_len = np.linalg.norm(vec)
        half_diff = (length - current_len) / 2.0
        direction = vec / current_len
        return entry - direction * half_diff, exit + direction * half_diff

    def generate_screw(self, entry: np.ndarray, exit: np.ndarray, output_path: str = "") -> np.ndarray:
        entry_adj, exit_adj = self.adjust_screw_to_length(entry, exit, self.total_length)
        screw_vec = exit_adj - entry_adj
        screw_len = np.linalg.norm(screw_vec)
        screw_dir = screw_vec / screw_len

        # Create an orthogonal vector
        ortho = np.cross(screw_dir, np.array([1, 0, 0]))
        if np.allclose(ortho, 0):
            ortho = np.cross(screw_dir, np.array([0, 1, 0]))
        ortho /= np.linalg.norm(ortho)

        angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        screw_coords = []

        for angle in angles:
            circle_offset = self.radius * (ortho * np.cos(angle) + np.cross(screw_dir, ortho) * np.sin(angle))
            for i in np.linspace(0, 1, self.num_points):
                point = entry_adj + i * screw_len * screw_dir + circle_offset
                screw_coords.append(point)

        screw_coords = np.array(screw_coords)
        if output_path:
            np.savetxt(output_path, screw_coords, delimiter=' ', fmt='%.3f')
            logging.info(f"Screw path saved to {output_path}")
        return screw_coords

    def process_screw_paths(self, input_folder: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
    
        for category in os.listdir(input_folder):
            category_path = os.path.join(input_folder, category)
            if not os.path.isdir(category_path):
                continue
    
            out_cat_path = os.path.join(output_folder, category)
            os.makedirs(out_cat_path, exist_ok=True)
    
            for file in os.listdir(category_path):
                if not file.endswith(".txt"):
                    continue
    
                file_path = os.path.join(category_path, file)
                try:
                    points = np.loadtxt(file_path, usecols=(1, 2, 3))
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
                    continue
    
                base = os.path.splitext(file)[0]
                if points.shape[0] == 2:
                    entry, exit = points[0], points[1]
                    coords = self.generate_screw(entry, exit)
                    output_path = os.path.join(out_cat_path, f"{base}_screw_1.txt")
                    np.savetxt(output_path, coords, fmt='%.3f')
                    logging.info(f"Screw generated for {base} (1)")
                elif points.shape[0] == 4:
                    for i in range(2):
                        entry, exit = points[i * 2], points[i * 2 + 1]
                        coords = self.generate_screw(entry, exit)
                        output_path = os.path.join(out_cat_path, f"{base}_screw_{i + 1}.txt")
                        np.savetxt(output_path, coords, fmt='%.3f')
                        logging.info(f"Screw {i + 1} generated for {base}")
                else:
                    logging.warning(f"Unexpected point count in {file_path} (found {points.shape[0]}), skipping.")
    
        logging.info("✅ All screw path generation completed.")

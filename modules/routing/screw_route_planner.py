# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/routing/screw_route_planner.py

import os
import numpy as np
import trimesh
import logging

logging.basicConfig(level=logging.INFO)

class ScrewRoutePlanner:
    def __init__(self):
        pass

    @staticmethod
    def find_intersections(start_point: np.ndarray, end_point: np.ndarray, mesh: trimesh.Trimesh):
        ray_origin = (start_point + end_point) / 2.0
        direction = end_point - start_point
        direction /= np.linalg.norm(direction)
        reverse = -direction

        hits_fwd, _, _ = mesh.ray.intersects_location([ray_origin], [direction])
        hits_bwd, _, _ = mesh.ray.intersects_location([ray_origin], [reverse])

        if len(hits_fwd) and len(hits_bwd):
            distances = [(p1, p2, np.linalg.norm(p1 - p2)) for p1 in hits_fwd for p2 in hits_bwd]
            best_pair = max(distances, key=lambda x: x[2])
            return best_pair[1], best_pair[0]
        elif len(hits_fwd):
            return hits_fwd[0], hits_fwd[0]
        elif len(hits_bwd):
            return hits_bwd[0], hits_bwd[0]
        return None, None

    def process_model(self, stl_file: str, points_file: str, output_dir: str):
        if not os.path.exists(points_file):
            logging.warning(f"Route points file {points_file} not found. Skipping.")
            return
    
        # Derive category (e.g., C1 or C2) from the points file name
        category = os.path.splitext(os.path.basename(points_file))[0]
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
    
        mesh = trimesh.load_mesh(stl_file)
    
        # Read route points from file and store as {label: point}
        points = {}
        with open(points_file) as f:
            for line in f:
                x, y, z, label = line.strip().split()
                points[label] = np.array([float(x), float(y), float(z)])
    
        # Define left and right screw path pairs
        left_pairs = [
            ('CrEntry_left', 'CrExit_left'),
            ('Entry_left', 'CrEntry_left'),
            ('CrEntry_left', 'Exit_left'),
            ('Entry_left', 'CrExit_left'),
            ('CrExit_left', 'Exit_left'),
            ('Entry_left', 'Exit_left')
        ]
        right_pairs = [
            ('CrEntry_right', 'CrExit_right'),
            ('Entry_right', 'CrEntry_right'),
            ('CrEntry_right', 'Exit_right'),
            ('Entry_right', 'CrExit_right'),
            ('CrExit_right', 'Exit_right'),
            ('Entry_right', 'Exit_right')
        ]
    
        # Use a dictionary to store the first valid results for each combination:
        # results = { combination_name: { 'left': (first, last), 'right': (first, last) } }
        results = {}
    
        # Process left pairs
        for start_label, end_label in left_pairs:
            if start_label not in points or end_label not in points:
                continue
            p_start, p_end = points[start_label], points[end_label]
            first, last = self.find_intersections(p_start, p_end, mesh)
            if first is None or last is None:
                continue
            combination_name = f"{start_label.split('_')[0]}_{end_label.split('_')[0]}"
            if combination_name not in results:
                results[combination_name] = {}
            # Store the left result if not already stored
            if 'left' not in results[combination_name]:
                results[combination_name]['left'] = (first, last)
    
        # Process right pairs
        for start_label, end_label in right_pairs:
            if start_label not in points or end_label not in points:
                continue
            p_start, p_end = points[start_label], points[end_label]
            first, last = self.find_intersections(p_start, p_end, mesh)
            if first is None or last is None:
                continue
            combination_name = f"{start_label.split('_')[0]}_{end_label.split('_')[0]}"
            if combination_name not in results:
                results[combination_name] = {}
            if 'right' not in results[combination_name]:
                results[combination_name]['right'] = (first, last)
    
        # For each combination that has both left and right results, write a file
        for combination_name, res in results.items():
            if 'left' in res and 'right' in res:
                left_first, left_last = res['left']
                right_first, right_last = res['right']
                output_text = (
                    f"first_left {left_first[0]:.6f} {left_first[1]:.6f} {left_first[2]:.6f}\n"
                    f"last_left {left_last[0]:.6f} {left_last[1]:.6f} {left_last[2]:.6f}\n"
                    f"first_right {right_first[0]:.6f} {right_first[1]:.6f} {right_first[2]:.6f}\n"
                    f"last_right {right_last[0]:.6f} {right_last[1]:.6f} {right_last[2]:.6f}\n"
                )
                out_file = os.path.join(category_output_dir, f"{combination_name}.txt")
                # Write (override) the file rather than appending
                with open(out_file, "w") as f:
                    f.write(output_text)
                logging.info(f"✅ Saved: {category}/{combination_name}.txt")
    
    


    def process_multiple_models(self, stl_files: list, points_folder: str, output_dir: str):
        for stl_file in stl_files:
            base_name = os.path.splitext(os.path.basename(stl_file))[0]
            points_path = os.path.join(points_folder, f"{base_name}.txt")
            self.process_model(stl_file, points_path, output_dir)

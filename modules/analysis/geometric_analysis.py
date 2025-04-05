# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/analysis/geometric_analysis.py

import os
import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.decomposition import PCA
import alphashape
import trimesh
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class GeometricAnalyzer:
    def __init__(self, valid_labels_c1=None, valid_labels_c2=None):
        self.valid_labels_c1 = valid_labels_c1 if valid_labels_c1 else [1, 2, 3, 4, 6, 7, 9, 10]
        self.valid_labels_c2 = valid_labels_c2 if valid_labels_c2 else [12, 13, 14, 15, 17, 18, 20, 21]

    @staticmethod
    def calculate_center(data: np.ndarray, mean: np.ndarray, label) -> np.ndarray:
        max_distances = np.array([np.max(np.linalg.norm(data - p, axis=1)) for p in data])
        sorted_data = data[max_distances.argsort()]
        top_points = sorted_data[:10] if len(sorted_data) > 9 else sorted_data
        return np.mean(top_points, axis=0)

    @staticmethod
    def compute_custom_centroids(df: pd.DataFrame, center_func, mean: np.ndarray, label) -> pd.DataFrame:
        centroids = {}
        for lbl, group in df.groupby('label'):
            data = group[['x', 'y', 'z']].to_numpy()
            center = center_func(data, mean, lbl)
            centroids[lbl] = center
        return pd.DataFrame.from_dict(centroids, orient='index', columns=['x', 'y', 'z'])

    @staticmethod
    def calculate_critical_center(data: np.ndarray, mean: np.ndarray, label) -> np.ndarray:
        X = data[:, 0]
        Y = data[:, 1]
        Z = data[:, 2]
        coordinates = np.vstack((X, Y, Z)).T
        alpha = 0.005

        alpha_shape = alphashape.alphashape(coordinates, alpha)
        vertices = np.array(alpha_shape.vertices)
        faces = np.array(alpha_shape.faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        pca = PCA(n_components=2)
        pca.fit(coordinates)
        projected_data = pca.transform(coordinates)
        normal = np.cross(pca.components_[0], pca.components_[1])

        def optimize_sphere(initial_guess):
            def objective(vars):
                radius = vars[0]
                center = vars[1:]
                if not mesh.contains([center])[0]:
                    return 1e6
                return -radius

            def plane_constraint(vars):
                center = vars[1:]
                return np.dot(center - initial_guess[1:], normal)

            def constraint_blue_points(vars):
                radius = vars[0]
                center = vars[1:]
                distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
                return np.min(distances) - radius

            def radius_constraint(vars):
                return 12 - vars[0]

            constraints = [
                {'type': 'ineq', 'fun': constraint_blue_points},
                {'type': 'eq', 'fun': plane_constraint},
                {'type': 'ineq', 'fun': radius_constraint}
                ]

            result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')
            optimal_radius = -result.fun
            optimal_center = result.x[1:]
            return optimal_radius, optimal_center

        np.random.seed(42)
        num_initial_guesses = min(20, len(vertices))
        random_indices = np.random.choice(len(vertices), size=num_initial_guesses, replace=False)
        initial_guesses = [
            np.hstack(([0.1], pca.inverse_transform(pca.transform([vertices[idx]]))[0]))
            for idx in random_indices
            ]

        results = []
        for guess in initial_guesses:
            radius, center = optimize_sphere(guess)
            if radius > 1:
                results.append((radius, center))

        top_20_spheres = sorted(results, key=lambda x: x[0], reverse=True)[:7]

        if label in [17, 18]:
            distances_to_mean = [np.linalg.norm(center - mean) for radius, center in top_20_spheres if radius > 1.5]
            closest_sphere_index = np.argmin(distances_to_mean) if distances_to_mean else 0
        else:
            closest_sphere_index = 0

        closest_sphere = top_20_spheres[closest_sphere_index]
        closest_radius, closest_center = closest_sphere
        #print(f"Critical center: {closest_center}, Radius: {closest_radius}")
        return closest_center


    @staticmethod
    def compute_centroids(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby('label')[['x', 'y', 'z']].mean()

    @staticmethod
    def read_point_cloud(file_path: str, valid_labels: list) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, names=['x', 'y', 'z', 'label'])
            return df[df['label'].isin(valid_labels)]
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def process_folder(self, data_folder: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
    
        category_to_centroids = {"C1": [], "C2": []}
    
        for root, _, files in os.walk(data_folder):
            for fname in files:
                file_path = os.path.join(root, fname)
    
                if not fname.endswith(".txt"):
                    continue
    
                # Check if file is related to C1 or C2
                category = "C1" if "_C1" in fname else "C2" if "_C2" in fname else None
                if category is None:
                    continue
    
                valid_labels = self.valid_labels_c1 if category == "C1" else self.valid_labels_c2
                df = self.read_point_cloud(file_path, valid_labels)
                if df.empty:
                    continue
    
                logging.info(f"Processing {fname}")
                mean_val = df[['x', 'y', 'z']].to_numpy().mean(axis=0)
    
                centroids_dict = {}
                for lbl in df['label'].unique():
                    group = df[df['label'] == lbl]
                    if lbl in [1, 2, 3, 4, 12, 13, 14, 15]:
                        cent = self.compute_custom_centroids(group, self.calculate_center, mean_val, lbl)
                    elif lbl in [9, 10]:
                        cent = self.compute_centroids(group)
                    else:
                        cent = self.compute_custom_centroids(group, self.calculate_critical_center, mean_val, lbl)
                    centroids_dict[lbl] = cent
    
                centroids_df = pd.concat(centroids_dict.values())
    
                def assign_region(index, region_name):
                    base = 'Entry_' if "EE" in fname and index in [1, 3, 12, 14] else \
                           'Exit_' if "EE" in fname else fname.split('_')[0] + '_'
                    return base + ('left' if index in [1, 2, 6, 9, 12, 13, 17, 20] else 'right')
    
                centroids_df['region'] = centroids_df.index.map(lambda idx: assign_region(idx, fname))
                centroids_df.reset_index(drop=True, inplace=True)
    
                # Keep only final output format: x y z region_label
                category_to_centroids[category].append(centroids_df[['x', 'y', 'z', 'region']])
    
        # Save as C1.txt and C2.txt
        for category, centroid_lists in category_to_centroids.items():
            if centroid_lists:
                final_df = pd.concat(centroid_lists)
                output_file = os.path.join(output_folder, f"{category}.txt")
                final_df.to_csv(output_file, sep=' ', header=False, index=False)
                logging.info(f"✅ Saved route points to {output_file}")
                print(f"✅ FINAL saved: {output_file}")

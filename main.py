# author Sanny Sahani, MCLAB, CGU, Taiwan
# main.py


import os
from modules.pointcloud.generate import PointCloudGenerator
from modules.processing.single_file_processor import SingleFileProcessor
from modules.alignment.pointcloud_alignment import PointCloudAlignment
from modules.segmentation.model_inference import SegmentationModel
from modules.analysis.geometric_analysis import GeometricAnalyzer
from modules.routing.screw_route_planner import ScrewRoutePlanner
from modules.path_generation.screw_path_planner import ScrewPathPlanner
from modules.visualization.viewer import launch_viewer

import numpy as np
import torch

def main():
    # === User Configuration ===
    input_models = {
        "subgl195_dirax_ct segmentation_C1 vertebra_lps.stl": "C1",
         "subgl195_dirax_ct segmentation_C2 vertebra_lps.stl": "C2",  # Add more if needed
    }
    model_log_dir = r"pretrained_models\pointnet2\log\part_seg\2024-10-14_18-50-pointnet2alinged"

    # === Ensure required directories ===
    os.makedirs("data/output_pointclouds", exist_ok=True)
    os.makedirs("data/segmented_regions", exist_ok=True)
    os.makedirs("data/route_points", exist_ok=True)
    os.makedirs("data/refined_routes", exist_ok=True)
    os.makedirs("data/screw_paths", exist_ok=True)

    # === Segmentation Model Load (once) ===
    model = SegmentationModel(log_dir=model_log_dir)

    for stl_name, category in input_models.items():
        stl_path = os.path.join("data/input_models", stl_name)
        base_name = os.path.splitext(stl_name)[0]
        pc_output_path = f"data/output_pointclouds/{base_name}.txt"

        print(f"\n🔍 Processing model: {stl_name} as {category}")

        # Step 1: Generate Point Cloud
        pc = PointCloudGenerator.process_model_to_point_cloud(stl_path, number_of_points=6000)
        PointCloudGenerator.save_point_cloud_with_normals(pc, pc_output_path)

        # Step 2: Align, Normalize & Sample
        pc_np = np.asarray(pc.points)
        processor = SingleFileProcessor(npoints=6000)
        sampled, centroid, m, transform = processor.process(pc_np, category)

        # Step 3: Segmentation
        print("🧠 Segmenting regions...")
        points_tensor = torch.tensor(sampled.T).unsqueeze(0).float().cuda()
        regions = model.run_segmentation_for_all_regions(points_tensor, category)

        # Step 4: Save region segmentations
        for region_name, labels in regions.items():
            coords = model.denormalize(sampled, centroid, m)
            coords = model.apply_inverse_transformation(coords, transform)
            save_path = f"data/segmented_regions/{region_name}_{category}.txt"
            with open(save_path, "w") as f:
                for point, label in zip(coords, labels):
                    f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f} {label}\n")
        print(f"✅ Saved region segmentations for {category}")

    # Step 5: Geometric Analysis (combine all segmented regions into C1.txt / C2.txt)
    print("📌 Extracting route points...")
    analyzer = GeometricAnalyzer()
    analyzer.process_folder("data/segmented_regions", "data/route_points")

    # Step 6: Raycasting for Screw Route Refinement
    print("🔦 Refining screw entry/exit...")
    screw_refiner = ScrewRoutePlanner()
    for stl_name, category in input_models.items():
        stl_path = os.path.join("data/input_models", stl_name)
        route_file = os.path.join("data/route_points", f"{category}.txt")
        if not os.path.exists(route_file):
            print(f"⚠️ Route file not found: {route_file}. Skipping.")
            continue
        screw_refiner.process_model(stl_path, route_file, "data/refined_routes")

    # Step 7: Screw Path Generation
    print("🔩 Generating screw path geometry...")
    screw_planner = ScrewPathPlanner()
    screw_planner.process_screw_paths("data/refined_routes", "data/screw_paths")

    # Step 8: Viewer
    print("🖼️ Launching 3D viewer...")
    stl_paths = [os.path.join("data/input_models", f) for f in input_models.keys()]
    screw_paths_root = "data/screw_paths"
    launch_viewer(stl_paths, screw_paths_root)
    


if __name__ == "__main__":
    main()

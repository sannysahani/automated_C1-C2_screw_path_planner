# 🧠 Automated C1–C2 Spine Screw Path Planning

This project provides a complete, modular pipeline for automated screw path planning using 3D STL models of the **C1–C2 cervical vertebrae**. It uses deep learning, point cloud processing, and anatomical geometry analysis to predict safe and reliable screw placements.

---

## 🚀 Key Features

✅ Load STL models of C1 and/or C2  
✅ Sample and align 3D point clouds  
✅ Segment anatomical regions using a pretrained PointNet++ model  
✅ Analyze and extract screw entry/exit route points  
✅ Refine screw paths via raycasting on 3D model surface  
✅ Visualize 3D vertebrae and screw paths interactively

---

## 🧰 Folder Structure

```
automated_C1-C2_screw_path_planner/
├── main.py                         # Main script (run this)
├── modules/                        # All pipeline logic (modular)
├── data/
│   ├── input_models/              # STL files of C1 and C2 vertebrae
│   ├── output_pointclouds/
│   ├── segmented_regions/
│   ├── route_points/
│   ├── refined_routes/
│   ├── standard_oreintation_data/
│   └── screw_paths/
├── pretrained_models/             # Pretrained PointNet++ logs + model
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/automated_C1-C2_screw_path_planner.git
cd automated_C1-C2_screw_path_planner
```

### 2. Create and Activate Environment

```bash
conda create -n spine_planner python=3.11
conda activate spine_planner
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔍 Inference Usage

1. Place STL files of C1 and/or C2 vertebrae into:
```
data/input_models/
```

2. Run the pipeline:
```bash
python main.py
```

3. Output will be saved to:
- Point clouds → `data/output_pointclouds/`
- Segmented regions → `data/segmented_regions/`
- Screw paths → `data/screw_paths/`
- 3D viewer will launch showing screw planning

---

## 🧠 Model Info

This project uses a pretrained **PointNet++ segmentation model** trained on vertebrae point cloud data. You can replace it with your own trained model or use our training pipeline (soon will be added in next version).

---

## 🧪 Sample Data

- Sample STL files are included in `data/input_models/`  
- You can replace them with your own C1 or C2 vertebrae STL files.

---

## 🔗 Citation

If you use this code, consider citing or linking to the GitHub repository and research paper.

---

## 📄 Related Publication

This project is based on the following research paper:

> **Automated Patient-Specific C1-C2 Posterior Cervical Fusion Screw Trajectory Planning using 3D Deep Learning**  
> Yau-Zen Chang, Sanny Kumar Sahani, and Chieh-Tsai Wu  
> Presented at International Joint Conference on Neural Networks (IJCNN), 2024.  
> [[PDF Download](docs/Automated_Patient-Specific_C1-C2_Posterior_Cervical_Fusion_Screw_Trajectory_Planning_using_3D_Deep_Learning.pdf)]

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

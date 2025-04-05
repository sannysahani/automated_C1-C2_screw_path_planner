# Sanny Sahani, MCLAB, CGU, Taiwan
# modules/segmentation/model_inference.py
import sys
sys.path.append("D:/Sanny MCLAB/automated_spine_screw_planner")
import os
import torch
import importlib
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
  # <-- Adjust to your project root

class SegmentationModel:
    def __init__(self, log_dir: str, num_part: int = 22, normal_channel: bool = False, gpu: str = '0'):
        self.logs_path = os.path.join(log_dir, "logs")
        self.log_dir = log_dir
        self.num_part = num_part
        self.normal_channel = normal_channel
        self.gpu = gpu
        self.classifier = self._load_model()

    def _load_model(self) -> torch.nn.Module:

        sys.path.append(self.logs_path)
        sys.path.append("C:/Users/MCLAB-01/Desktop/pointnet++")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        logs_path = os.path.join(self.log_dir, 'logs')


        model_files = [f for f in os.listdir(logs_path) if f.endswith('.py')]
        if not model_files:
            raise FileNotFoundError("No model files found in logs directory.")
        
        model_name = model_files[0].replace('.py', '')
        MODEL = importlib.import_module(model_name)

        classifier = MODEL.get_model(self.num_part, normal_channel=self.normal_channel).cuda()
        checkpoint_path = os.path.join(logs_path, 'checkpoints', 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        logging.info("Model loaded successfully.")
        return classifier

    @staticmethod
    def to_categorical(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
        return new_y.cuda() if y.is_cuda else new_y

    def run_segmentation_inference_for_label(self, points_tensor: torch.Tensor, dummy_label: int,
                                             num_votes: int = 3, num_classes: int = 16) -> np.ndarray:
        vote_pool = torch.zeros(points_tensor.size(0), points_tensor.size(2), self.num_part).cuda()
        dummy_tensor = torch.tensor([dummy_label], dtype=torch.long).cuda()

        for _ in range(num_votes):
            seg_pred, _ = self.classifier(points_tensor, self.to_categorical(dummy_tensor, num_classes))
            vote_pool += seg_pred

        seg_pred = vote_pool / num_votes
        predicted_labels = torch.argmax(seg_pred, dim=2).squeeze(0).cpu().numpy()
        return predicted_labels

    def run_segmentation_for_all_regions(self, points_tensor: torch.Tensor, category: str,
                                         num_votes: int = 3, num_classes: int = 16) -> dict:
        if category == 'C1':
            dummy_labels = {'EE_region': 0, 'CrEntry_region': 1, 'CrExit_region': 2}
        elif category == 'C2':
            dummy_labels = {'EE_region': 3, 'CrEntry_region': 4, 'CrExit_region': 5}
        else:
            logging.error("Invalid category. Must be 'C1' or 'C2'.")
            return None

        region_results = {}
        for region, label in dummy_labels.items():
            preds = self.run_segmentation_inference_for_label(points_tensor, label, num_votes, num_classes)
            region_results[region] = preds
        return region_results

    @staticmethod
    def denormalize(pc: np.ndarray, centroid: np.ndarray, m: float) -> np.ndarray:
        return pc * m + centroid

    @staticmethod
    def apply_inverse_transformation(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        transformation_inv = np.linalg.inv(transformation_matrix)
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        reverted = points_homogeneous @ transformation_inv.T
        return reverted[:, :3]

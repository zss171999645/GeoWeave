import numpy as np

def matrix_to_similarity3(mat: np.ndarray):
    from gtsam import Rot3, Point3, Similarity3
    """Convert 4x4 Sim3 matrix to gtsam.Similarity3"""
    R = mat[:3, :3]
    scale = np.cbrt(np.linalg.det(R))  # 取立方根作为尺度
    t = mat[:3, 3] / scale
    R_unit = R / scale
    return Similarity3(Rot3(R_unit), Point3(*t), scale)

import pickle
import os

class SerializableMixin:
    def save(self, filepath):
        """Serialize the object's member variables to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filepath):
        """Deserialize member variables from a file into this object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)
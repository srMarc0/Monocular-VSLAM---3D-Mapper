import os
import cv2
import numpy as np

class ImageDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.images_info = self._load_images_info()
        self.gt_poses = self._load_ground_truth()
        self.index = 0

    def _load_images_info(self):
        images = []
        path = os.path.join(self.base_path, "info/images.txt")
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                # Store: image_id, timestamp, full_path
                images.append({
                    'id': int(parts[0]),
                    'path': os.path.join(self.base_path, parts[2])
                })
        return images

    def _load_ground_truth(self):
        gt = {}
        path = os.path.join(self.base_path, "info/groundtruth.txt")
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("#"): continue
                p = [float(x) for x in line.split()]
                # ID: [tx, ty, tz, qx, qy, qz, qw]
                gt[int(p[0])] = np.array(p[1:])
        return gt

    def read(self):
        if self.index >= len(self.images_info):
            return None, None, None

        info = self.images_info[self.index]
        img = cv2.imread(info['path'])
        gt = self.gt_poses.get(info['id'])
        
        self.index += 1
        return img, gt, info['id']
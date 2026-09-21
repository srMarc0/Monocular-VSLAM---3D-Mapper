import numpy as np

def calculate_error(est, gt):
    est, gt = np.array(est), np.array(gt)
    # Ensure they are the same length
    min_len = min(len(est), len(gt))
    rmse = np.sqrt(np.mean(np.sum((est[:min_len] - gt[:min_len])**2, axis=1)))
    print(f"--- EVALUATION ---")
    print(f"Absolute Trajectory Error (RMSE): {rmse:.4f} units")
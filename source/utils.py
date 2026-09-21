import cv2
import numpy as np

def get_camera_matrix(mode, width=None, height=None):
    """Returns the Intrinsic Camera Matrix (K) based on the mode"""
    if mode == "dataset":
        # Exact hardcoded K for dataset mode
        K = np.array([[329.115520046, 0, 320.0],
                      [0, 329.115520046, 240.0],
                      [0, 0, 1]], dtype=np.float64)
        return K
    elif mode == "video":
        # Dynamically estimates K for video mode
        focal_length = float(width)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[focal_length, 0, cx],
                      [0, focal_length, cy],
                      [0, 0, 1]], dtype=np.float64)
        return K
    else:
        return np.eye(3, dtype=np.float64)

def read_frame(cap):
    ret, frame = cap.read()
    
    if not ret: return None
    return frame

def detect_features(frame):
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    
    return keypoints, descriptors

def match_features(current_frame, previous_frame):
    if current_frame is None or previous_frame is None: return []
    
    # matcher per a descriptors binaris (ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(current_frame, previous_frame, k=2)
    
    good_matches = []
    for m_n in knn_matches:
        if len(m_n) == 2:
            m, n = m_n
            # Lowe's Ratio Test to filter out ambiguous ORB matches
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        elif len(m_n) == 1:
            good_matches.append(m_n[0])
            
    matches = sorted(good_matches, key=lambda x: x.distance)
    
    return matches

def estimate_motion(kp1, kp2, matches, K):
    """
    kp1: keypoints del frame anterior
    kp2: keypoints del frame actual
    matches: llista de matches trobats
    K: Matriu de la càmera
    """
    if len(matches) < 8:  # Necessitem com a mínim 8 punts per la Matriu Fonamental
        return None, None, None

    # Convertim els keypoints a llistes de punts (x, y)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None or E.shape != (3, 3):
        return None, None, None

    # Recuperem la Rotació i la Translació a partir de la Matriu Essencial
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    return R, t, mask_pose

def triangulate_points(R, t, kp1, kp2, matches, K): #seria el parallax (més o menys jo confio)
    # Definim la matriu de projecció de la primera càmera (estàtica a l'origen)
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    
    # Matriu de la segona càmera (la nova posició): P2 = K * [R | t]
    P2 = np.dot(K, np.hstack((R, t)))

    # Extraure punts dels matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

    # Triangulació (retorna punts en 4D: X, Y, Z, W)
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    
    # Convertir de 4D a 3D (dividir per W)
    pts3D = pts4D[:3] / pts4D[3]
    
    return pts3D.T # Retorna llista de punts (X, Y, Z)


def quaternion_to_rotation_matrix(q):
    """Converteix un quaternó [qx, qy, qz, qw] a una matriu de rotació 3x3"""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    
def local_bundle_adjustment(trajectory, pts3D_list, current_R, current_t, K, deviation_thresh=0.6):
    """
    Ajust local progressiu i suau. 
    Predeix la posició actual de la càmera basant-se en el vector de velocitat dels darrers frames
    i corregeix la posició cap a un punt intermedi si es desvia massa.
    """
    if len(trajectory) < 8 or len(pts3D_list) == 0:
        return current_R, current_t

    # 1. Predicció geomètrica de la posició de la càmera (extrapolació lineal)
    # Agafem la posició de fa 2 frames actius (per veure la tendència del vector de moviment)
    p_ant2 = np.array(trajectory[-3])
    p_ant1 = np.array(trajectory[-1])
    
    # Vector de moviment estimat recentment (direcció i velocitat simulada)
    velocity_vector = p_ant1 - p_ant2
    predicted_t = p_ant1 + velocity_vector

    # 2. Comparem la posició que ens dóna el frame actual (current_t) amb la predicció
    actual_t = current_t.flatten()
    deviation = np.linalg.norm(actual_t - predicted_t)

    # 3. Si la desviació supera el llindar, fem un reajustament suau a un terme mitjà
    if deviation > deviation_thresh:
        # Factor d'interpolació (0.5 significa exactament al mig entre la predicció i el càlcul actual)
        alpha = 0.4 
        
        # Corregim la translació actual de forma suau
        corrected_t = (1 - alpha) * actual_t + alpha * predicted_t
        current_t = corrected_t.reshape(3, 1)
        
        # Opcional: Podríem moure lleugerament els punts 3D del món per coherència,
        # però corregint la càmera ja evitem que el mapa es trenqui en els següents frames.

    return current_R, current_t


def export_to_ply(filepath, points, colors=None):
    """
    Exporta una llista de punts 3D a un fitxer .ply
    points: array Nx3 o llista de numpy arrays (N, 3)
    colors: array Nx3 o llista de numpy arrays (N, 3) amb valors de 0 a 255. Opcional.
    """
    points = np.array(points).reshape(-1, 3)
    if colors is not None:
        colors = np.array(colors).reshape(-1, 3).astype(np.uint8)
    
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            if colors is not None:
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(line + "\n")

def smooth_trajectory(trajectory, pts3D_list, current_R, current_t, deviation_thresh=0.05):
    """
    Estima la posició de la càmera basant-se en el vector de velocitat dels darrers frames
    i corregeix la posició cap a un punt intermedi si es desvia massa.
    """
    if len(trajectory) < 8 or len(pts3D_list) == 0:
        return current_R, current_t

    # 1. Predicció geomètrica de la posició de la càmera (extrapolació lineal)
    # Agafem la posició de fa 2 frames actius (per veure la tendència del vector de moviment)
    p_ant2 = np.array(trajectory[-3])
    p_ant1 = np.array(trajectory[-1])
    
    # Vector de moviment estimat recentment (direcció i velocitat simulada)
    velocity_vector = p_ant1 - p_ant2
    predicted_t = p_ant1 + velocity_vector

    # 2. Comparem la posició que ens dóna el frame actual (current_t) amb la predicció
    actual_t = current_t.flatten()
    deviation = np.linalg.norm(actual_t - predicted_t)

    # 3. Si la desviació supera el llindar, fem un reajustament suau a un terme mitjà
    if deviation > deviation_thresh:
        # Factor d'interpolació (0.4 significa un ajuste equilibrado entre la predicción y el cálculo)
        alpha = 0.4 
        
        # Corregim la translació actual de la càmera
        corrected_t = (1 - alpha) * actual_t + alpha * predicted_t
        current_t = corrected_t.reshape(3, 1)
        
    return current_R, current_t
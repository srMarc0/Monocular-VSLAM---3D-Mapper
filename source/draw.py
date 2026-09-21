import cv2
import numpy as np

def visualize_matches(prev_frame, prev_kp, curr_frame, curr_kp, matches, max_width=1280):
    img_matches = cv2.drawMatches(
        prev_frame, prev_kp, 
        curr_frame, curr_kp, 
        matches[:50], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    h, w = img_matches.shape[:2]

    cv2.putText(img_matches, "ANTERIOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_matches, "ACTUAL", (w//2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Barra d'instruccions inferior
    cv2.rectangle(img_matches, (0, h - 35), (w, h), (0, 0, 0), -1)
    instruccions = "[SPC] Pausa/Pas a pas | [Q] Sortir"
    cv2.putText(img_matches, instruccions, (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Redimensionat automàtic
    if w > max_width:
        ratio = max_width / float(w)
        dim = (max_width, int(h * ratio))
        img_matches = cv2.resize(img_matches, dim, interpolation=cv2.INTER_AREA)

    return img_matches


def draw_trajectory(curr_t, canvas):
    """
    Dibuixa la trajectòria de la càmera en un mapa 2D.
    t: vector de translació actual acumulat
    """
    # Escalam i desplacem el punt perquè es vegi al centre del llenç
    # Nota: El factor 50 és per fer el moviment més visible (zoom)
    x = int(curr_t[0]) + 300
    z = int(curr_t[2]) + 300 # Usem Z per a la profunditat (mapa vist des de dalt)

    # Dibuixem un cercle petit de color verd per a la posició actual
    cv2.circle(canvas, (x, z), 1, (0, 255, 0), 2)
    
    return canvas

obs_layer = np.zeros((600, 600, 3), dtype=np.uint8)
traj_layer = np.zeros((600, 600, 3), dtype=np.uint8)
gt_layer = np.zeros((600, 600, 3), dtype=np.uint8)

last_x_world = None
last_z_world = None
shifted_x = 0   # píxels enters ja aplicats a les capes (acumulat)
shifted_z = 0

def draw_map(curr_t, gt_normalized, points3D):
    global obs_layer, traj_layer, gt_layer
    global last_x_world, last_z_world, shifted_x, shifted_z

    scale = 2
    center_x = 300
    center_z = 300

    curr_x_world = curr_t[0].item() * scale
    curr_z_world = -curr_t[1].item() * scale

    if last_x_world is not None:
        target_x = int(round(curr_x_world))
        target_z = int(round(curr_z_world))

        dx = target_x - shifted_x
        dz = target_z - shifted_z

        if dx != 0 or dz != 0:
            M = np.float32([[1, 0, -dx], [0, 1, -dz]])
            obs_layer  = cv2.warpAffine(obs_layer,  M, (600, 600), flags=cv2.INTER_NEAREST)
            traj_layer = cv2.warpAffine(traj_layer, M, (600, 600), flags=cv2.INTER_NEAREST)
            gt_layer   = cv2.warpAffine(gt_layer,   M, (600, 600), flags=cv2.INTER_NEAREST)
            shifted_x += dx
            shifted_z += dz

    last_x_world = curr_x_world
    last_z_world = curr_z_world

    # Posició de la càmera "arrodonida" que coincideix amb el desplaçament de les capes
    ref_x = shifted_x
    ref_z = shifted_z

    # 1. Punts 3D
    for global_pt in points3D:
        x_obj = int(round(center_x + global_pt[0] * scale - ref_x))
        z_obj = int(round(center_z - global_pt[1] * scale - ref_z))
        if 0 <= x_obj < 600 and 0 <= z_obj < 600:
            cv2.circle(obs_layer, (x_obj, z_obj), 1, (80, 80, 80), -1)

    # 2. Trajectòria estimada (posició real respecte a la referència entera)
    x_cam = int(round(center_x + curr_x_world - ref_x))
    z_cam = int(round(center_z + curr_z_world - ref_z))
    cv2.circle(traj_layer, (x_cam, z_cam), 2, (0, 255, 0), -1)

    # 3. Ground truth
    if gt_normalized is not None:
        x_gt = int(round(center_x + gt_normalized[0].item() * scale - ref_x))
        z_gt = int(round(center_z - gt_normalized[1].item() * scale - ref_z))
        if 0 <= x_gt < 600 and 0 <= z_gt < 600:
            cv2.circle(gt_layer, (x_gt, z_gt), 1, (0, 0, 255), -1)

    res = cv2.add(obs_layer, traj_layer)
    res = cv2.add(res, gt_layer)

    cv2.putText(res, "Green: Estimated", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(res, "Red: Ground Truth", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return res

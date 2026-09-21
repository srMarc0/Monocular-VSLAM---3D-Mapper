import cv2
import numpy as np
from source.utils import *
from source.draw import *
from source.dataset import ImageDataset
from source.metrics import *
from scipy.spatial.transform import Rotation as Rot 
import time 
		
def main(mode="dataset", video_path=None):
    """
    mode: "dataset" per utilitzar les imatges RPG Urban Pinhole amb GT.
          "video" per utilitzar un vídeo casolà (ex: mp4).
    """
    start = time.time()
    if mode == "dataset":
        dataset = ImageDataset("dataset/rpg_urban_pinhole_data/data")
    elif mode == "video":
        if video_path is None:
            print("Error: Has de proporcionar la ruta d'un vídeo per al mode 'video'.")
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No es pot obrir el vídeo a: {video_path}")
            return
    else:
        print("Error: Mode no vàlid. Tria 'dataset' o 'video'.")
        return

    last_frame_data = None
    cur_R = None             # S'inicialitzarà dinàmicament al primer frame
    cur_t = np.zeros((3, 1)) # Comencem a (0,0,0) del món
    gt_offset = None
    K = None                 # Matriu de càmera dinàmica

    est_trajectory = []
    gt_trajectory = []

    # ----- PERFORMANCE TUNING PARAMETERS -----
    FRAME_SKIP = 2          # process every 2rd frame (set to 0 to process all)
    frame_counter = 0
    
    all_pts3d = []
    all_colors3d = []
    
    print(f"--- Iniciant Visual SLAM en mode: {mode.upper()} ---")
    
    while True:
        pts3D_world = []
        # -- LECTURA DE FRAMES SEGONS EL MODE
        if mode == "dataset":
            frame, gt_pose, img_id = dataset.read()
            if frame is None: break
        else: # mode == "video"
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            gt_pose = None

        if K is None:
            K = get_camera_matrix(mode, width=frame.shape[1], height=frame.shape[0])

        # -- LÒGICA DE FRAME_SKIP (SARTAR FRAMES PER RENDIMENT)
        if FRAME_SKIP > 0 and frame_counter % (FRAME_SKIP + 1) != 0:
            if mode == "dataset" and gt_pose is not None:
                if gt_offset is None:
                    gt_offset = gt_pose[:3].copy()
                    cur_R = Rot.from_quat(gt_pose[3:]).as_matrix()
                gt_normalized = gt_pose[:3] - gt_offset
                if last_frame_data is not None and cur_t is not None:
                    est_trajectory.append(cur_t.flatten())
                    gt_trajectory.append(gt_normalized.flatten())
            elif mode == "video":
                if cur_R is None:
                    cur_R = np.eye(3) # Càmera inicial mirant endavant si salta el primer frame
                if last_frame_data is not None and cur_t is not None:
                    est_trajectory.append(cur_t.flatten())
            
            frame_counter += 1
            continue
            
        #--- CONFIGURACIÓ DE L'ORIENTACIÓ INICIAL I GROUND TRUTH
        if mode == "dataset" and gt_pose is not None:
            if gt_offset is None:
                gt_offset = gt_pose[:3].copy()
                cur_R = Rot.from_quat(gt_pose[3:]).as_matrix()
            gt_normalized = gt_pose[:3] - gt_offset
        else:
            gt_normalized = None
            
            if cur_R is None:
                cur_R = np.eye(3)

        # -- EXTRACCIÓ DE CARACTERÍSTIQUES
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = detect_features(gray_frame)

        #qTRACKING I ESTIMACIÓ GEOMÈTRICA
        if last_frame_data is not None and des is not None:
            matches = match_features(last_frame_data['descriptors'], des)  # Associar
            
            if len(matches) >= 8:
                R, t, mask = estimate_motion(last_frame_data['keypoints'], kp, matches, K)
            
            if t is not None and mask is not None:
                # Filtre RANSAC per netejar falsos positius
                inlier_matches = [matches[i] for i in range(len(matches)) if mask[i][0] > 0]
                
                # Càlcul o assignació de l'escala de translació
                if mode == "dataset":
                    if gt_pose is not None and last_frame_data['gt'] is not None:
                        prev_gt = last_frame_data['gt']
                        curr_gt = gt_pose[:3]
                        absolute_scale = np.linalg.norm(curr_gt - prev_gt)
                        valid_step = (absolute_scale > 0.01)
                    else:
                        absolute_scale = 1.0
                        valid_step = False
                elif mode == "video":
                    absolute_scale = 1.0
                    valid_step = True
                else:
                    absolute_scale = 1.0
                    valid_step = False

                if valid_step and len(inlier_matches) > 8:
                    # Guardem la pose de la càmera anterior per a la triangulació
                    prev_R = cur_R.copy()
                    prev_t = cur_t.copy()

                    # Acumulació del moviment global de la càmera
                    cur_R = cur_R @ R.T
                    cur_t = cur_t + absolute_scale * (-cur_R @ t)

                    # Triangulació estricta d'inliers amb l'escala real aplicada
                    scaled_t = t * absolute_scale
                    new_pts3D = triangulate_points(R, scaled_t, last_frame_data['keypoints'], kp, inlier_matches, K)
                    
                    # Statistical Outlier Removal based on Depth (Z)
                    if len(new_pts3D) > 0:
                        depths = np.array([pt[2] for pt in new_pts3D])
                        mean_z = np.mean(depths)
                        std_z = np.std(depths)
                    else:
                        mean_z, std_z = 0, 0

                    # Filtre de profunditat local (Z) abans de passar-ho al món global
                    pts3D_world = []
                    for i, pt in enumerate(new_pts3D):
                        z = pt[2]
                        # Retain points with Z in (0.5, 200.0) and within 2 std deviations
                        if 0.5 < z < 200.0 and abs(z - mean_z) < 2 * std_z:
                            # Projectem usant la pose de la càmera ANTERIOR on s'ha originat la matriu P1 de triangulació
                            pt_w = (prev_R @ pt.reshape(3, 1) + prev_t).flatten()
                            pts3D_world.append(pt_w)
                            
                            idx = inlier_matches[i].trainIdx
                            pt_2d = kp[idx].pt
                            x, y = int(pt_2d[0]), int(pt_2d[1])
                            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                                color = frame[y, x]
                            else:
                                color = [128, 128, 128]
                            # OpenCV uses BGR, so we convert to RGB for PLY
                            all_colors3d.append([color[2], color[1], color[0]])
                            all_pts3d.append(pt_w)
                else:
                    pts3D_world = []

                # Guardem dades per a mètriques
                est_trajectory.append(cur_t.flatten())
                if mode == "dataset" and gt_normalized is not None:
                    gt_trajectory.append(gt_normalized.flatten())

                # Dibuixem el mapa de capes (si és vídeo, gt_normalized serà None de forma segura)
                gt_param = gt_normalized.reshape(3, 1) if gt_normalized is not None else None
                map_fused = draw_map(cur_t, gt_param, pts3D_world)
                cv2.imshow('Mapa 2D - SLAM (Capes)', map_fused)
                
                # --- OPTIMITZACIÓ LOCAL CADA 8 FRAMES --- #
                if frame_counter % 8 == 0 and len(est_trajectory) > 8:
                    cur_R, cur_t = local_bundle_adjustment(
                        est_trajectory, pts3D_world, cur_R, cur_t,
                        K=K,
                        deviation_thresh=0.6
                    )

                
            # Visualització del seguiment de punts en temps real
            display_img = visualize_matches(
                last_frame_data['frame'], last_frame_data['keypoints'], 
                frame, kp, matches
            )
            cv2.imshow('Visual SLAM - Tracking', display_img)
        else:
            cv2.imshow('Visual SLAM - Tracking', frame)

        # Actualitzem l'estat per al següent frame
        last_frame_data = {
            'keypoints': kp, 
            'descriptors': des, 
            'frame': frame.copy(),
            'gt': gt_pose[:3] if gt_pose is not None else None,
            'pts3D': pts3D_world
        }
        frame_counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): 
            print("Pausa activa. Prem qualsevol tecla per continuar...")
            cv2.waitKey(0)    
    
    # --- Tancar video --- 
    if mode == "video":
        cap.release()

    # --- Càlcul error amb dataset ---
    if mode == "dataset" and len(gt_trajectory) > 0:
        calculate_error(est_trajectory, gt_trajectory)
        
    # --- Exportar representació 3D ---			
    if len(all_pts3d) > 0:
        print(f"Exportant representació 3D ({len(all_pts3d)} punts)...")
        export_to_ply("map 3D/map_3d.ply", all_pts3d, all_colors3d)
        print("S'ha desat el núvol de punts a 'map_3d.ply'. Pots obrir-lo amb MeshLab o un altre visor 3D.")
        
    end = time.time()
    print("Temps:", end - start)

    print("Processament acabat. Prem 'Q' en qualsevol finestra per tancar el programa.")
    while True:
        end_key = cv2.waitKey(10) & 0xFF
        if end_key == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Opció A: Mode Dataset Urbà (Amb Ground Truth i errors)
    main(mode="dataset")
    
    # Opció B: Mode Vídeo Casolà (Sense Ground Truth, escala simulada)
    #main(mode="video", video_path="videos/v1.mp4")
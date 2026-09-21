# Mapejador d'escenaris (Visual SLAM Monocular)

Projecte desenvolupat per a les assignatures de **Visió per Computador (VC)** i **Processament de Senyals i Imatge Digital (PSIV)** de l'Escola d'Enginyeria (EE) de la **Universitat Autònoma de Barcelona (UAB)**.

---

## 📌 Descripció del Projecte
Aquest projecte consisteix en el disseny i desenvolupament d'una aplicació de **Visió per Computador** focalitzada en el mapatge i la navegació en entorns mitjançant tècniques de **Visual SLAM monocular**. 

L'objectiu principal és transformar la informació captada per una única càmera en moviment en un mapa detallat que permeti estimar la trajectòria de la càmera, diferenciar entre zones de lliure circulació i identificar obstacles.

---

## 🛠️ Tecnologies i Tècniques Utilitzades
* **Detecció de característiques:** ORB (Oriented FAST and Rotated BRIEF).
* **Seguiment i Matching:** Lucas-Kanade Optical Flow i correspondència de punts entre frames.
* **Estimació de moviment:** Triangulació de punts 2D per obtenir coordenades 3D i càlcul de la matriu essencial.
* **Optimització global:** Bundle Adjustment per corregir la deriva (*drift*).
* **Avaluació de rendiment:** Mètrica d'Error Quadràtic Mitjà (MSE) en comparació amb el *Ground Truth*.

---

## 🔄 Flux del Sistema
1. **Captura i preprocessament:** Conversió de frames a escala de grisos.
2. **Detecció i Matching:** Extracció de punts claus amb ORB i aparellament entre el frame actual i l'anterior.
3. **Odometria Visual:** Càlcul de la rotació i translació de la càmera.
4. **Triangulació i Mapatge:** Reconstrucció espacial i actualització de la posició global.
5. **Visualització en temps real:** 
   * Monitor de seguiment de correspondències visuals.
   * Mapa 2D/3D amb la trajectòria estimada vs. *Ground Truth*.

---

## 📊 Datasets i Proves
Per avaluar el sistema s'han utilitzat dues fonts de dades:
1. **Vídeos propis/casolans:** Gravacions en entorns interiors (passadissos i habitacions) per a l'avaluació qualitativa de girs i canvis d'il·luminació.
2. **Dataset Sintètic (Blender) / Tercers:** Utilització de la seqüència sintètica creada amb **Blender** per *Zhang et al.* per poder avaluar quantitativament l'error de localització (MSE) respecte a un *Ground Truth* conegut.

---

## 👥 Autors
* **Marc Solés i Rojas**
* **Samuel Jesus Outeda Aponte**
* **Felipe Martín Marcano Hurtado**

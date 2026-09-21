# Mapejador d'Escenaris — Monocular Visual SLAM

Sistema de visió per computador per al mapatge 2D/3D i la navegació en entorns interiors utilitzant una única font de vídeo (Visió Monocular) i tècniques de **Visual SLAM**.

> **Projecte acadèmic:** Visió per Computador (VC) - Escola d'Enginyeria, UAB.  
> **Autors:** Marc Solés i Rojas, Samuel Jesus Outeda Aponte, Felipe Martín Marcano Hurtado.

---

## 🚀 Característiques Principals

- **SLAM Monocular:** Reconstrucció de l'entorn i estimació del moviment a partir d'una sola càmera.
- **Detecció de Característiques:** Extracció de punts clau i *feature matching* entre fotogrames mitjançant l'algorisme **ORB**.
- **Estimació de Moviment:** Càlcul d'odometria visual, matriu essencial i triangulació de punts 3D.
- **Visualització en Temps Real:**
  - Finestra de *tracking* amb la correspondència de punts entre frames consecutius.
  - Generació del mapa i trajectòria (estimada vs. *Ground Truth*).
- **Avaluació Mètrica:** Càlcul de l'error acumulat (MSE) comparant amb dades de referència (*Ground Truth*).

---

## 🛠️ Tecnologia i Algorismes

- **Processament d'imatge:** Escala de grisos i extracció de punts característics (ORB / SIFT / SURF).
- **Seguiment:** Flux òptic (Lucas-Kanade) i *Feature Matching*.
- **Optimització:** Bundle Adjustment per reduir la deriva (*drift*).

---

## 📊 Flux de Treball

1. **Captura:** Lectura de frames seqüencials (vídeo propi o *dataset*).
2. **Preprocessament:** Conversió a escala de grisos i definició de les condicions inicials de la càmera.
3. **Tracking & Matching:** Extracció de punts amb ORB i cerca de correspondències.
4. **Odometria & Triangulació:** Estimació de la rotació ($R$) i translació ($T$) per resoldre la posició 3D dels punts.
5. **Ajust d'Escala & Mètriques:** Calibració d'escala absoluta mitjançant *Ground Truth* i càlcul de l'error quadrat mitjà (MSE).

---

## 🔬 Resultats i Mètriques

El sistema permet avaluar:
- **Error de localització:** Precisió de la trajectòria estimada respecte al mapa real.
- **Robustesa:** Comportament davant de girs ràpids o canvis d'il·luminació.
- **Cost computacional:** Temps de processament per frame.

---

## 🔮 Futures Millores

- Transició completa de la visualització de 2D a una reconstrucció 3D densa.
- Millora en la gestió dels girs bruscos de la càmera per evitar pèrdues de seguiment.
- Integració de la segmentació semàntica per distingir àrees transitables d'obstacles mòbils.
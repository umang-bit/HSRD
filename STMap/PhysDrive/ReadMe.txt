# PhysDrive STMap Generation
# Adapted from HSRD/STMap/BUAA pipeline for the PhysDrive dataset.
# PhysDrive videos are 30 FPS constant, so no CSI resampling is needed.

Handling procedure:
1. Landmark.py:  Detect 68 face landmarks from RGB.mp4 using face_alignment → Label/RGB_lmk.csv
2. Align_Face.py: 3-point affine face alignment (128x128), mask out eyes/nose/mouth → Align/
3. STMap.py:     Generate STMap_RGB.png from aligned faces (5x5 grid = 25 ROIs) → STMap/STMap_RGB.png

Run all steps:
  sbatch run_stmap_pipeline.sbatch

Or run individually:
  python Landmark.py --gpu 0                    # all subjects
  python Landmark.py --gpu 0 --subject AFH1     # single subject
  python Align_Face.py                           # all subjects
  python Align_Face.py --subject AFH1            # single subject
  python STMap.py                                # all subjects
  python STMap.py --subject AFH1                 # single subject

Each step skips sessions that are already processed (checks for existing output files).

"""
Step 2: Face alignment using 68 landmarks.
Reads RGB.mp4 + Label/RGB_lmk.csv, applies:
  - Face skin mask (exclude eyes, nose, mouth using landmark polygons)
  - 3-point affine alignment to 128x128 (landmarks 1, 15, 8 → canonical positions)
Saves aligned frames to {subject}/{session}/Align/{frame_id}.png

Usage:
  python Align_Face.py --subject AFH1       # process single subject
  python Align_Face.py                       # process all subjects
"""

import os
import argparse
import csv
import cv2
import copy
import numpy as np

DATASET_ROOT = "/scratch/umang.tiwari/datasets/PhysDrive(Publication)"


def get_subject_list(worker_id=None, num_workers=None, single_subject=None):
    """Get list of subjects for this worker."""
    subjects = sorted([s for s in os.listdir(DATASET_ROOT)
                       if os.path.isdir(os.path.join(DATASET_ROOT, s))])
    if single_subject:
        return [single_subject]
    if worker_id is not None and num_workers is not None:
        subjects = [s for i, s in enumerate(subjects) if i % num_workers == worker_id]
    return subjects


def process_session(sess_path):
    """Align faces for one session using pre-computed landmarks."""
    video_path = os.path.join(sess_path, 'Video', 'RGB.mp4')
    lmk_path = os.path.join(sess_path, 'Label', 'RGB_lmk.csv')
    align_path = os.path.join(sess_path, 'Align')

    if not os.path.exists(video_path):
        print(f"  SKIP: no RGB.mp4")
        return False
    if not os.path.exists(lmk_path):
        print(f"  SKIP: no RGB_lmk.csv (run Landmark.py first)")
        return False

    # Read landmarks
    lmk_all = []
    with open(lmk_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lmk_all.append(line)

    os.makedirs(align_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lmk_index = 0
    frame_id = 10000  # same numbering convention as HSRD

    while cap.isOpened():
        ret, img_temp = cap.read()
        if not ret:
            break
        if lmk_index >= len(lmk_all):
            break

        lmk = np.array(lmk_all[lmk_index], dtype=np.float32).reshape(-1, 2)
        lmk_index += 1

        # Skip frames where no face was detected (all zeros)
        if np.sum(np.abs(lmk)) < 1e-6:
            frame_id += 1
            continue

        # --- Create face skin mask (exclude eyes, nose, mouth) ---
        h, w = img_temp.shape[:2]
        Mask1 = np.zeros_like(img_temp, dtype="uint8")
        Mask2 = np.zeros_like(img_temp, dtype="uint8")
        Mask3 = np.zeros_like(img_temp, dtype="uint8")
        Mask4 = np.zeros_like(img_temp, dtype="uint8")
        Mask5 = np.zeros_like(img_temp, dtype="uint8")

        # Face contour
        ROI1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 22, 21, 17]
        ROI2 = [36, 17, 21, 39, 40, 41]  # left eye
        ROI3 = [42, 22, 26, 45, 46, 47]  # right eye
        ROI4 = [31, 33, 35, 30]           # nose
        ROI5 = [51, 53, 54, 55, 57, 59, 48, 49]  # mouth

        def roi_pts(indices):
            return [[round(float(lmk[i, 0])), round(float(lmk[i, 1]))] for i in indices]

        cv2.fillPoly(Mask1, np.int32([roi_pts(ROI1)]), (255, 255, 255))
        cv2.fillPoly(Mask2, np.int32([roi_pts(ROI2)]), (255, 255, 255))
        cv2.fillPoly(Mask3, np.int32([roi_pts(ROI3)]), (255, 255, 255))
        cv2.fillPoly(Mask4, np.int32([roi_pts(ROI4)]), (255, 255, 255))
        cv2.fillPoly(Mask5, np.int32([roi_pts(ROI5)]), (255, 255, 255))

        # Subtract eyes/nose/mouth from face mask
        cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask2), Mask1)
        cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask3), Mask1)
        cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask4), Mask1)
        cv2.bitwise_and(Mask1, cv2.bitwise_not(Mask5), Mask1)

        img_Masked = cv2.bitwise_and(img_temp, Mask1)

        # --- 3-point affine alignment to 128x128 ---
        # Landmarks: 1 (left jaw) → (0, 48), 15 (right jaw) → (128, 48), 8 (chin) → (64, 128)
        old = np.array([
            [lmk[1, 0], lmk[1, 1]],
            [lmk[15, 0], lmk[15, 1]],
            [lmk[8, 0], lmk[8, 1]]
        ], np.float32)
        new = np.array([
            [0, 48],
            [128, 48],
            [64, 128]
        ], np.float32)

        M = cv2.getAffineTransform(old, new)
        Face_align = cv2.warpAffine(img_temp, M, (img_temp.shape[1], img_temp.shape[0]))

        # Crop to 128x128
        out_path = os.path.join(align_path, str(frame_id) + '.png')
        cv2.imwrite(out_path, Face_align[0:128, 0:128, :], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        frame_id += 1

        if (lmk_index) % 500 == 0:
            print(f"    Frame {lmk_index}/{total_frames}")

    cap.release()
    saved_count = frame_id - 10000
    print(f"  Saved {saved_count} aligned frames")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=None, help='Process single subject')
    parser.add_argument('--worker_id', type=int, default=None, help='Worker index (0-based)')
    parser.add_argument('--num_workers', type=int, default=None, help='Total number of workers')
    args = parser.parse_args()

    subjects = get_subject_list(args.worker_id, args.num_workers, args.subject)
    print(f"Processing {len(subjects)} subjects: {subjects}")

    total = 0
    for subj in subjects:
        subj_path = os.path.join(DATASET_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue
        for sess in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, sess)
            if not os.path.isdir(sess_path):
                continue

            align_path = os.path.join(sess_path, 'Align')
            # Skip if already processed
            if os.path.isdir(align_path) and len(os.listdir(align_path)) > 0:
                print(f"SKIP {subj}/{sess} (Align/ already exists)")
                continue

            print(f"Processing {subj}/{sess}")
            process_session(sess_path)
            total += 1

    print(f"\nDone! Processed {total} sessions.")


if __name__ == '__main__':
    main()

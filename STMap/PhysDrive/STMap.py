"""
Step 3: Generate STMap from aligned face images.
Reads Align/ folder + Label/RGB_lmk.csv, divides each face into 5x5 grid (25 ROIs),
computes mean BGR per ROI per frame, normalizes, saves STMap_RGB.png.

PhysDrive is at 30 FPS constant, so no CSI resampling is needed (unlike VIPL).

Usage:
  python STMap.py --subject AFH1      # process single subject
  python STMap.py                      # process all subjects
"""

import os
import argparse
import math
import csv
import cv2
import numpy as np


def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], value_type=1, lmk_type=2):
    """
    Extract mean BGR from face sub-regions.
    value_type=1: Simple 5x5 grid on already-aligned image (no landmark rotation needed).
    value_type=2: Landmark-based rotation + crop + 5x5 grid.
    """
    Value = []
    h, w, c = img.shape
    if value_type == 1:
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = img[h_index * h_step:(h_index + 1) * h_step,
                           w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.nanmean(np.nanmean(temp, axis=0), axis=0)
                Value.append(temp1)
    elif value_type == 2:
        lmk = np.array(lmk, np.float32).reshape(-1, 2)
        min_p = np.min(lmk, 0)
        max_p = np.max(lmk, 0)
        min_p = np.maximum(min_p, 0)
        max_p = np.minimum(max_p, [w - 1, h - 1])

        left_eye = lmk[36:41]
        right_eye = lmk[42:47]
        left = np.array([lmk[0], lmk[1], lmk[2]])
        right = np.array([lmk[14], lmk[15], lmk[16]])

        left_eye = np.nanmean(left_eye, 0)
        right_eye = np.nanmean(right_eye, 0)
        left = np.nanmean(left, 0)
        right = np.nanmean(right, 0)
        top = max((left[1] + right[1]) / 2 - 0.5 * (max_p[1] - (left[1] + right[1]) / 2), 0)
        rotate_angular = math.atan(
            (right_eye[1] - left_eye[1]) / (0.00001 + right_eye[0] - left_eye[0])
        ) * (180 / math.pi)

        cent_point = [w / 2, h / 2]
        matRotation = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angular, 1)
        face_rotate = cv2.warpAffine(img, matRotation, (w, h))
        left[0], left[1] = PointRotate(math.radians(rotate_angular), left[0], left[1],
                                        cent_point[0], cent_point[1])
        right[0], right[1] = PointRotate(math.radians(rotate_angular), right[0], right[1],
                                          cent_point[0], cent_point[1])
        face_crop = face_rotate[int(top):int(max_p[1]), int(left[0]):int(right[0]), :]
        h, w, c = face_crop.shape
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = face_crop[h_index * h_step:(h_index + 1) * h_step,
                                 w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)


def mySTMap(imglist_root, lmk_all=[]):
    """
    Generate STMap from aligned face images.
    Since images are already 3-point aligned to 128x128, we use type=1 (simple 5x5 grid).
    """
    img_list = sorted(os.listdir(imglist_root))
    STMap = []
    z = 0
    for fname in img_list:
        fpath = os.path.join(imglist_root, fname)
        img = cv2.imread(fpath)
        if img is None:
            z += 1
            continue
        # type=1: images are already aligned, just use 5x5 grid
        Value = getValue(img, lmk=lmk_all[z] if z < len(lmk_all) else [], value_type=1)
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value)
        z += 1

    STMap = np.array(STMap)  # [num_frames, 25, 3]

    # Normalize per-ROI per-channel to [0, 255]
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            col = STMap[:, w, c]
            STMap[:, w, c] = 255 * ((col - np.nanmin(col)) / (
                0.001 + np.nanmax(col) - np.nanmin(col)))

    STMap = np.swapaxes(STMap, 0, 1)  # [25, num_frames, 3]
    STMap = np.rint(STMap)
    STMap = np.array(STMap, dtype='uint8')
    return STMap


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
            stmap_dir = os.path.join(sess_path, 'STMap')
            stmap_path = os.path.join(stmap_dir, 'STMap_RGB.png')

            if not os.path.isdir(align_path) or len(os.listdir(align_path)) == 0:
                print(f"SKIP {subj}/{sess} (no Align/ folder — run Align_Face.py first)")
                continue

            if os.path.exists(stmap_path):
                print(f"SKIP {subj}/{sess} (STMap_RGB.png exists)")
                continue

            # Read landmarks (not strictly needed for type=1, but kept for compatibility)
            lmk_path = os.path.join(sess_path, 'Label', 'RGB_lmk.csv')
            lmk_all = []
            if os.path.exists(lmk_path):
                with open(lmk_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    for line in reader:
                        lmk_all.append(line)

            print(f"Generating STMap for {subj}/{sess} ({len(os.listdir(align_path))} frames)")
            stmap = mySTMap(align_path, lmk_all=lmk_all)

            os.makedirs(stmap_dir, exist_ok=True)
            cv2.imwrite(stmap_path, stmap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"  Saved {stmap_path} (shape: {stmap.shape})")
            total += 1

    print(f"\nDone! Generated {total} STMaps.")


if __name__ == '__main__':
    main()

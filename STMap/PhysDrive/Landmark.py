"""
Step 1: Detect 68 face landmarks from RGB.mp4 videos using face_alignment.
Saves landmarks to {subject}/{session}/Label/RGB_lmk.csv

Optimization: Detect face bounding box once per video, reuse for all frames.
This gives ~10x speedup over per-frame face detection.

Multi-GPU: Use --worker_id and --num_workers to split subjects across GPUs.

Usage:
  python Landmark.py --gpu 0                                    # all subjects, 1 GPU
  python Landmark.py --gpu 0 --subject AFH1                     # single subject
  python Landmark.py --gpu 0 --worker_id 0 --num_workers 4      # parallel: worker 0 of 4
"""

import cv2
import os
import argparse
import numpy as np
import face_alignment
import csv
import time

DATASET_ROOT = "/scratch/umang.tiwari/datasets/PhysDrive(Publication)"


def process_video(video_path, save_path, fa):
    """Detect 68 landmarks per frame, save as CSV (136 values per row: x0,y0,x1,y1,...).

    Detects face bounding box ONCE on the first frame, then reuses for all frames.
    This gives ~13 fps vs ~1 fps with per-frame detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lmk = []
    frame_idx = 0
    face_bbox = None
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face bounding box only on the first frame
        if face_bbox is None:
            det = fa.face_detector.detect_from_image(frame)
            if len(det) > 0:
                face_bbox = det[0][:4].tolist()

        # Get landmarks using cached bounding box (skips expensive face detection)
        if face_bbox is not None:
            preds = fa.get_landmarks(frame, detected_faces=[face_bbox])
        else:
            preds = fa.get_landmarks(frame)

        if preds is None:
            lmk.append([0 for _ in range(136)])
        else:
            lmk.append(preds[0].reshape(136).tolist())

        frame_idx += 1
        if frame_idx % 1000 == 0:
            elapsed = time.time() - t0
            fps = frame_idx / elapsed
            eta = (total_frames - frame_idx) / fps
            print(f"    Frame {frame_idx}/{total_frames} ({fps:.1f} fps, ETA {eta:.0f}s)")

    cap.release()

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, 'RGB_lmk.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in lmk:
            writer.writerow(row)

    elapsed = time.time() - t0
    print(f"  Saved {len(lmk)} landmarks to {csv_path} ({elapsed:.1f}s)")
    return True


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
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--subject', type=str, default=None, help='Process single subject')
    parser.add_argument('--worker_id', type=int, default=None, help='Worker index (0-based)')
    parser.add_argument('--num_workers', type=int, default=None, help='Total number of workers')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
    )

    subjects = get_subject_list(args.worker_id, args.num_workers, args.subject)
    print(f"Processing {len(subjects)} subjects: {subjects}")

    total = 0
    for subj in subjects:
        subj_path = os.path.join(DATASET_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue
        sessions = sorted(os.listdir(subj_path))
        for sess in sessions:
            sess_path = os.path.join(subj_path, sess)
            if not os.path.isdir(sess_path):
                continue

            save_path = os.path.join(sess_path, 'Label')
            # Skip if landmarks already exist
            if os.path.exists(os.path.join(save_path, 'RGB_lmk.csv')):
                print(f"SKIP {subj}/{sess} (RGB_lmk.csv exists)")
                continue

            video_path = os.path.join(sess_path, 'Video', 'RGB.mp4')
            if not os.path.exists(video_path):
                print(f"SKIP {subj}/{sess} (no RGB.mp4)")
                continue

            print(f"Processing {subj}/{sess}")
            process_video(video_path, save_path, fa)
            total += 1

    print(f"\nDone! Processed {total} sessions.")


if __name__ == '__main__':
    main()

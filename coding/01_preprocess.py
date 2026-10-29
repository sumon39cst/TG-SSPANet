"""Convert the Figshare brain-tumor .mat files into a single compressed NumPy cache.

Output: E:/Life_Nero/dataset/processed/figshare_{SIZE}.npz with
    images  uint8  (N, S, S)   min-max normalised slice
    masks   uint8  (N, S, S)   binary tumour mask (tumorMask field)
    labels  int64  (N,)        0=meningioma, 1=glioma, 2=pituitary
    pids    <U16   (N,)        patient id
    ids     int64  (N,)        original file index (1..3064)
    cvfold  int64  (N,)        official patient-wise 5-fold index (1..5) from cvind.mat
"""
import os
import glob
import h5py
import numpy as np
import cv2

ROOT = r"E:\Life_Nero\dataset"
SIZE = 256


def read_mat(path):
    with h5py.File(path, "r") as f:
        c = f["cjdata"]
        img = np.array(c["image"], dtype=np.float32).T
        mask = np.array(c["tumorMask"], dtype=np.uint8).T
        label = int(np.array(c["label"]).ravel()[0])
        pid = "".join(chr(int(v)) for v in np.array(c["PID"]).ravel())
    return img, mask, label, pid


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "raw", "*.mat")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    with h5py.File(os.path.join(ROOT, "cvind.mat"), "r") as f:
        cvind = np.array(f["cvind"]).ravel().astype(np.int64)
    assert len(files) == len(cvind) == 3064, (len(files), len(cvind))

    images, masks, labels, pids, ids = [], [], [], [], []
    for p in files:
        img, mask, label, pid = read_mat(p)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255.0
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        images.append(img.astype(np.uint8))
        masks.append((mask > 0).astype(np.uint8))
        labels.append(label - 1)
        pids.append(pid)
        ids.append(int(os.path.splitext(os.path.basename(p))[0]))

    out = os.path.join(ROOT, "processed")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"figshare_{SIZE}.npz"),
                        images=np.stack(images), masks=np.stack(masks),
                        labels=np.array(labels, dtype=np.int64), pids=np.array(pids),
                        ids=np.array(ids, dtype=np.int64), cvfold=cvind)
    labels = np.array(labels)
    print("saved", len(images), "slices; class counts", np.bincount(labels),
          "; patients", len(set(pids)), "; folds", np.bincount(cvind)[1:])


if __name__ == "__main__":
    main()

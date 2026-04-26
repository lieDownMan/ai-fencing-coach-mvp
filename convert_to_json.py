#!/usr/bin/env python3
"""
convert_to_json.py — 將 FFD/ 資料夾中的 mp4 影片轉換成 FencingDataset 所需的 JSON 格式。
                     同時進行資料增強 (翻轉、雜訊、時間扭曲)。
                     (已升級使用 YOLOv8-Pose 引擎)

用法:
    python convert_to_json.py

輸出:
    data/json_samples/ 資料夾下的 JSON 檔案，每支影片產生多個增強版本。
    可直接用於: python train_fencenet.py --data_dir data/json_samples
"""

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import interp1d

# ================= 設定 =================

FFD_DIR = "./FFD"                    # 來源資料夾
OUTPUT_DIR = "./data/json_samples"   # 輸出 JSON 的資料夾

# 資料夾名稱 → FencingDataset 的 label 對應
FOLDER_TO_LABEL = {
    "0_SF":   "SF",
    "1_SB":   "SB",
    "2_R":    "R",
    "3_IS":   "IS",   
    "4_WW":   "WW",
    "5_JS":   "JS",
}
FENCER_ID_DEFAULT = "fencer_01"

# YOLOv8 (COCO 格式) 關節索引 → FencingDataset 期望的命名對應
# 假設選手的「前手」(持劍手) 是右手
YOLO_JOINT_MAP = {
    "nose":             0,
    "front_wrist":      10,  # right_wrist
    "front_elbow":      8,   # right_elbow
    "front_shoulder":   6,   # right_shoulder
    "front_ankle":      16,  # right_ankle
    "left_hip":         11,
    "right_hip":        12,
    "left_knee":        13,
    "right_knee":       14,
    "left_ankle":       15,
    "right_ankle":      16,
}

# 關節名稱的順序 (用於 numpy 轉換)
JOINT_NAMES = list(YOLO_JOINT_MAP.keys())

# 增強參數
NOISE_LEVEL = 0.015
MAX_WARP_RATIO = 0.15

# ================= 初始化 YOLOv8 =================
print("正在載入 YOLOv8-Pose 模型...")
model = YOLO('yolov8n-pose.pt') # 首次執行會自動下載輕量級權重
print("模型載入完成！")


# ================= 格式轉換工具 =================

def keypoints_to_numpy(keypoints):
    T = len(keypoints)
    J = len(JOINT_NAMES)
    arr = np.zeros((T, J, 2), dtype=np.float64)
    for t, frame in enumerate(keypoints):
        for j, name in enumerate(JOINT_NAMES):
            arr[t, j, 0] = frame[name][0]
            arr[t, j, 1] = frame[name][1]
    return arr


def numpy_to_keypoints(arr):
    T = arr.shape[0]
    keypoints = []
    for t in range(T):
        frame = {}
        for j, name in enumerate(JOINT_NAMES):
            frame[name] = [round(float(arr[t, j, 0]), 6), round(float(arr[t, j, 1]), 6)]
        keypoints.append(frame)
    return keypoints


# ================= 資料增強函數 =================

def horizontal_flip(arr):
    flipped = np.copy(arr)
    flipped[:, :, 0] = -flipped[:, :, 0]

    swap_pairs = [
        ("left_hip",   "right_hip"),
        ("left_knee",  "right_knee"),
        ("left_ankle", "right_ankle"),
    ]

    for left_name, right_name in swap_pairs:
        li = JOINT_NAMES.index(left_name)
        ri = JOINT_NAMES.index(right_name)
        temp = np.copy(flipped[:, li, :])
        flipped[:, li, :] = flipped[:, ri, :]
        flipped[:, ri, :] = temp

    return flipped


def add_gaussian_noise(arr, noise_level=NOISE_LEVEL):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=arr.shape)
    return arr + noise


def time_warp(arr, max_warp_ratio=MAX_WARP_RATIO):
    T, J, C = arr.shape
    warp_ratio = np.random.uniform(1.0 - max_warp_ratio, 1.0 + max_warp_ratio)
    new_T = max(2, int(T * warp_ratio))

    orig_steps = np.linspace(0, 1, T)
    new_steps = np.linspace(0, 1, new_T)

    flat = arr.reshape(T, -1) 
    interpolator = interp1d(orig_steps, flat, axis=0, kind='linear')
    warped = interpolator(new_steps)

    if new_T < T:
        pad = np.tile(warped[-1:], (T - new_T, 1))
        warped = np.vstack((warped, pad))
    else:
        warped = warped[:T]

    return warped.reshape(T, J, C)


# ================= 核心提取函數 (YOLOv8 升級版) =================

def extract_keypoints_from_video(video_path):
    """
    從影片中提取每一幀的命名關節座標 (使用 YOLOv8 的正規化 xyn 座標)。
    回傳: list of dict，每個 dict 是一幀的關節座標。
    """
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 可以直接吃 BGR 格式，並回傳預測結果 (verbose=False 關閉終端機洗版)
        results = model(frame, verbose=False)

        # 確認有偵測到人，且 keypoints 有資料
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # 取得正規化座標 (xyn: X, Y Normalized)，等同於 MediaPipe 的輸出比例 0.0~1.0
            keypoints = results[0].keypoints.xyn[0].cpu().numpy()
            
            frame_dict = {}
            for joint_name, yolo_idx in YOLO_JOINT_MAP.items():
                kx, ky = keypoints[yolo_idx]
                frame_dict[joint_name] = [round(float(kx), 6), round(float(ky), 6)]
            all_frames.append(frame_dict)
        else:
            # 偵測失敗 → 用上一幀填補，或補零
            if len(all_frames) > 0:
                all_frames.append(all_frames[-1].copy())
            else:
                zero_frame = {name: [0.0, 0.0] for name in YOLO_JOINT_MAP.keys()}
                all_frames.append(zero_frame)

    cap.release()
    return all_frames


def save_json(keypoints, label, fencer_id, output_path):
    sample = {
        "label": label,
        "fencer_id": fencer_id,
        "keypoints": keypoints,
    }
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)


# ================= 主程式 =================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_converted = 0
    total_skipped = 0

    print(f"🚀 開始掃描 {FFD_DIR} 並轉換成 JSON (含資料增強) ...")
    print(f"📂 輸出目錄: {OUTPUT_DIR}")
    print(f"🔄 增強策略: 原始 + 翻轉 + 雜訊 + 時間扭曲 + 翻轉雜訊 + 翻轉時間扭曲 = 6 倍\n")

    for folder_name in sorted(os.listdir(FFD_DIR)):
        folder_path = os.path.join(FFD_DIR, folder_name)

        if not os.path.isdir(folder_path) or folder_name not in FOLDER_TO_LABEL:
            if os.path.isdir(folder_path) and folder_name not in FOLDER_TO_LABEL:
                print(f"⏭️  跳過 [{folder_name}] (不在分類對應表中)")
            continue

        label = FOLDER_TO_LABEL[folder_name]
        video_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mp4")])

        print(f"📁 處理 [{folder_name}] → label=\"{label}\"  ({len(video_files)} 支影片)")

        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            base_name = os.path.splitext(video_file)[0]

            keypoints = extract_keypoints_from_video(video_path)

            if len(keypoints) == 0:
                print(f"   ⚠️ {video_file}: 沒有偵測到任何骨架，跳過")
                total_skipped += 1
                continue

            arr = keypoints_to_numpy(keypoints)

            versions = {
                "orig":         arr,
                "flip":         horizontal_flip(arr),
                "noise":        add_gaussian_noise(arr),
                "twarp":        time_warp(arr),
                "flip_noise":   add_gaussian_noise(horizontal_flip(arr)),
                "flip_twarp":   time_warp(horizontal_flip(arr)),
            }

            for suffix, aug_arr in versions.items():
                aug_keypoints = numpy_to_keypoints(aug_arr)
                json_filename = f"{label}_{base_name}_{suffix}.json"
                json_path = os.path.join(OUTPUT_DIR, json_filename)
                save_json(aug_keypoints, label, FENCER_ID_DEFAULT, json_path)

            count = len(versions)
            total_converted += count
            print(f"   ✅ {video_file} → {count} 個 JSON  ({len(keypoints)} 幀)")

    print(f"\n{'='*50}")
    print(f"🎉 轉換完成！")
    print(f"   ✅ 總共產生: {total_converted} 個 JSON 檔案")
    print(f"   ⚠️ 跳過: {total_skipped} 支影片")
    print(f"   📂 JSON 檔案位於: {OUTPUT_DIR}")
    print(f"\n下一步: python train_fencenet.py --data_dir {OUTPUT_DIR}")
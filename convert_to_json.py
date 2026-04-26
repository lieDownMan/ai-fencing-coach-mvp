#!/usr/bin/env python3
"""
convert_to_json.py — 將 FFD/ 資料夾中的 mp4 影片轉換成 FencingDataset 所需的 JSON 格式。
                     同時進行資料增強 (翻轉、雜訊、時間扭曲)。

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
import mediapipe as mp
from scipy.interpolate import interp1d

# ================= 設定 =================

FFD_DIR = "./FFD"                    # 來源資料夾
OUTPUT_DIR = "./data/json_samples"   # 輸出 JSON 的資料夾

# 資料夾名稱 → FencingDataset 的 label 對應
# FencingDataset CLASS_NAMES = ["R", "IS", "WW", "JS", "SF", "SB"]
FOLDER_TO_LABEL = {
    "0_SF":   "SF",
    "1_SB":   "SB",
    "2_R":    "R",
    "3_IS":   "IS",   
    "4_WW":   "WW",
    "5_JS":   "JS",
    # "6_Idle" 不在六種動作分類中，跳過
}
# fencer_id：目前全部設成同一人，如果有多個選手請手動修改
FENCER_ID_DEFAULT = "fencer_01"

# MediaPipe 關節索引 → FencingDataset 期望的命名對應
# 假設選手的「前手」(持劍手) 是右手
MP_JOINT_MAP = {
    "nose":             0,
    "front_wrist":      16,  # right_wrist
    "front_elbow":      14,  # right_elbow
    "front_shoulder":   12,  # right_shoulder
    "front_ankle":      28,  # right_ankle (僅用於 normalization scale)
    "left_hip":         23,
    "right_hip":        24,
    "left_knee":        25,
    "right_knee":       26,
    "left_ankle":       27,
    "right_ankle":      28,
}

# 關節名稱的順序 (用於 numpy 轉換)
JOINT_NAMES = list(MP_JOINT_MAP.keys())

# 增強參數
NOISE_LEVEL = 0.015
MAX_WARP_RATIO = 0.15

# ================= 初始化 MediaPipe =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ================= 格式轉換工具 =================

def keypoints_to_numpy(keypoints):
    """
    list[dict] → numpy array, shape (T, J, 2)
    J = len(JOINT_NAMES), 2 = (x, y)
    """
    T = len(keypoints)
    J = len(JOINT_NAMES)
    arr = np.zeros((T, J, 2), dtype=np.float64)
    for t, frame in enumerate(keypoints):
        for j, name in enumerate(JOINT_NAMES):
            arr[t, j, 0] = frame[name][0]
            arr[t, j, 1] = frame[name][1]
    return arr


def numpy_to_keypoints(arr):
    """
    numpy array (T, J, 2) → list[dict]
    """
    T = arr.shape[0]
    keypoints = []
    for t in range(T):
        frame = {}
        for j, name in enumerate(JOINT_NAMES):
            frame[name] = [round(float(arr[t, j, 0]), 6), round(float(arr[t, j, 1]), 6)]
        keypoints.append(frame)
    return keypoints


# ================= 資料增強函數 (改寫自 build_dataset.py) =================

def horizontal_flip(arr):
    """
    水平翻轉骨架 (T, J, 2)：
    1. 將所有 x 座標取反 (模擬鏡像)
    2. 對調左右肢體的命名關節
    """
    flipped = np.copy(arr)

    # 1. 翻轉 x 座標
    flipped[:, :, 0] = -flipped[:, :, 0]

    # 2. 對調左右肢體 (在 JOINT_NAMES 中的索引)
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
    """
    加入高斯雜訊 (T, J, 2)
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=arr.shape)
    return arr + noise


def time_warp(arr, max_warp_ratio=MAX_WARP_RATIO):
    """
    時間扭曲 (T, J, 2)：隨機拉伸/壓縮時間軸，再對齊回原本幀數。
    """
    T, J, C = arr.shape
    warp_ratio = np.random.uniform(1.0 - max_warp_ratio, 1.0 + max_warp_ratio)
    new_T = max(2, int(T * warp_ratio))

    orig_steps = np.linspace(0, 1, T)
    new_steps = np.linspace(0, 1, new_T)

    # 對每個 (joint, coord) 通道做插值
    flat = arr.reshape(T, -1)  # (T, J*2)
    interpolator = interp1d(orig_steps, flat, axis=0, kind='linear')
    warped = interpolator(new_steps)  # (new_T, J*2)

    # 對齊回原本幀數
    if new_T < T:
        pad = np.tile(warped[-1:], (T - new_T, 1))
        warped = np.vstack((warped, pad))
    else:
        warped = warped[:T]

    return warped.reshape(T, J, C)


# ================= 核心提取函數 =================

def extract_keypoints_from_video(video_path):
    """
    從影片中提取每一幀的命名關節座標 (raw x, y)。
    回傳: list of dict，每個 dict 是一幀的關節座標。
    """
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_dict = {}
            for joint_name, mp_idx in MP_JOINT_MAP.items():
                lm = landmarks[mp_idx]
                frame_dict[joint_name] = [round(lm.x, 6), round(lm.y, 6)]
            all_frames.append(frame_dict)
        else:
            # 偵測失敗 → 用上一幀填補，或補零
            if len(all_frames) > 0:
                all_frames.append(all_frames[-1].copy())
            else:
                zero_frame = {name: [0.0, 0.0] for name in MP_JOINT_MAP.keys()}
                all_frames.append(zero_frame)

    cap.release()
    return all_frames


def save_json(keypoints, label, fencer_id, output_path):
    """儲存一個 JSON 樣本"""
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

            # 提取骨架
            keypoints = extract_keypoints_from_video(video_path)

            if len(keypoints) == 0:
                print(f"   ⚠️ {video_file}: 沒有偵測到任何骨架，跳過")
                total_skipped += 1
                continue

            # 轉成 numpy 做增強
            arr = keypoints_to_numpy(keypoints)  # (T, J, 2)

            # --- 產生 6 個版本 ---
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

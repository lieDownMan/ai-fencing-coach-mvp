import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SELECTED_JOINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32, 16] 

def extract_continuous_pose(video_path):
    """讀取整支長影片，回傳連續的骨架矩陣與原始骨盆 x 座標"""
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    pelvis_x_data = []  # 記錄每幀原始骨盆 x 座標（用來判斷移動方向）
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            frame_points = []
            landmarks = results.pose_landmarks.landmark
            pelvis_x = (landmarks[23].x + landmarks[24].x) / 2
            pelvis_y = (landmarks[23].y + landmarks[24].y) / 2
            
            for idx in SELECTED_JOINTS:
                norm_x = landmarks[idx].x - pelvis_x
                norm_y = landmarks[idx].y - pelvis_y
                frame_points.extend([norm_x, norm_y])
            frames_data.append(frame_points)
            pelvis_x_data.append(pelvis_x)
        else:
            if len(frames_data) > 0:
                frames_data.append(frames_data[-1])
                pelvis_x_data.append(pelvis_x_data[-1])
            else:
                frames_data.append([0.0] * (len(SELECTED_JOINTS) * 2))
                pelvis_x_data.append(0.5)  # 預設值（畫面中間）
                
    cap.release()
    return np.array(frames_data), np.array(pelvis_x_data)

def get_action_indices(frames_data, pelvis_x_data, threshold=0.04, direction_filter=True):
    """
    找出動作的起始與結束「幀數索引 (Frame Index)」
    direction_filter=True 時，第一個動作照抓，之後只保留從右到左的動作
    """
    diff = np.abs(np.diff(frames_data, axis=0))
    movement = np.sum(diff, axis=1)
    smoothed_movement = np.convolve(movement, np.ones(5)/5, mode='same')
    
    # 尋找移動量曲線的波峰
    peaks, _ = find_peaks(smoothed_movement, distance=30, height=threshold)
    
    indices = []
    first_captured = False
    
    for peak in peaks:
        start_idx = max(0, peak - 30)
        end_idx = min(len(frames_data), peak + 30)
        
        if not first_captured:
            # 第一個動作無條件保留
            indices.append((start_idx, end_idx))
            first_captured = True
            continue
        
        if direction_filter:
            # 判斷動作方向：比較 peak 前後的骨盆 x 座標
            # 注意：影片座標系中 x 從左到右遞增
            # 「從右到左」= pelvis_x 在 peak 附近遞減
            look_back = max(0, peak - 15)
            look_ahead = min(len(pelvis_x_data) - 1, peak + 15)
            
            x_before = pelvis_x_data[look_back]
            x_after = pelvis_x_data[look_ahead]
            
            # 影片被旋轉 180 度，所以實際的「右到左」在原始座標中是 x 遞增
            # 如果你的影片沒有旋轉，改成 x_before > x_after
            if x_before < x_after:
                # 右到左的動作（考慮 180 度旋轉後）
                indices.append((start_idx, end_idx))
                print(f"  ➡️⬅️ 保留動作 (peak={peak}): 骨盆 x {x_before:.3f} -> {x_after:.3f} (右→左)")
            else:
                print(f"  ⬅️➡️ 跳過動作 (peak={peak}): 骨盆 x {x_before:.3f} -> {x_after:.3f} (左→右)")
        else:
            indices.append((start_idx, end_idx))
        
    return indices

def export_video_clips(video_path, indices, output_dir="output_clips"):
    """
    根據給定的幀數範圍，將原影片切割並輸出為多個短影片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # 防呆機制
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 使用 mp4v 編碼匯出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i, (start, end) in enumerate(indices):
        out_path = os.path.join(output_dir, f"action_{i+1:03d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        # 將讀取頭跳到指定的起始幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        frames_to_read = end - start
        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_180)
            out.write(frame)
            
        out.release()
        print(f"✅ 已匯出: {out_path} (範圍: 第 {start} 幀 ~ 第 {end} 幀)")
        
    cap.release()

# ================= 測試執行區 =================
if __name__ == "__main__":
    # ⚠️ 記得把這裡換成你實際的長影片路徑！
    video_file = "./IMG_1622.mov" 
    
    if os.path.exists(video_file):
        print(f"正在分析影片: {video_file} ...")
        
        # 1. 抓取連續骨架
        continuous_data, pelvis_x_data = extract_continuous_pose(video_file)
        
        # 2. 計算出動作在哪幾幀發生 (修改 threshold 來微調靈敏度)
        print("正在計算動作波峰...")
        action_indices = get_action_indices(continuous_data, pelvis_x_data, threshold=0.04)
        print(f"程式判定總共有 {len(action_indices)} 個動作。")
        
        # 3. 實際切出影片來看看
        print("開始匯出影片片段...")
        export_video_clips(video_file, action_indices, output_dir="./4_WW")
        
        print("🎉 全部完成！請去 my_action_clips 資料夾檢查影片切得準不準。")
    else:
        print(f"❌ 找不到影片檔案: {video_file}，請確認路徑是否正確。")
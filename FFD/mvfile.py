import os
import shutil
import re

src_dir = "2_R"
dst_sb = "tmp"   # odd-numbered actions
dst_sf = "2_R"   # even-numbered actions

# 確保目標資料夾存在
os.makedirs(dst_sb, exist_ok=True)
os.makedirs(dst_sf, exist_ok=True)

pattern = re.compile(r"action_(\d+)\.mp4")

moved_sb = 0
moved_sf = 0

for filename in sorted(os.listdir(src_dir)):
    match = pattern.match(filename)
    if not match:
        continue

    action_num = int(match.group(1))
    src_path = os.path.join(src_dir, filename)

    if action_num % 2 == 1:  # odd -> 1_SB
        dst_path = os.path.join(dst_sb, filename)
        shutil.move(src_path, dst_path)
        moved_sb += 1
        print(f"✅ {filename} -> {dst_sb}/")
    else:  # even -> 0_SF
        dst_path = os.path.join(dst_sf, filename,)
        shutil.move(src_path, dst_path)
        moved_sf += 1
        print(f"✅ {filename} -> {dst_sf}/")

print(f"\n完成！移動了 {moved_sb} 個檔案到 {dst_sb}，{moved_sf} 個檔案到 {dst_sf}")

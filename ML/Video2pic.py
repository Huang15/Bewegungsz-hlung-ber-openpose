import cv2
import os

def extract_frames_from_videos(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有视频文件
    for video_filename in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_filename)
        
        # 检查是否为视频文件（可以根据需要添加更多的扩展名）
        if not video_filename.endswith(('.avi', '.mp4', '.mov', '.mkv')):
            continue
        
        # 为每个视频文件创建一个单独的输出文件夹
        video_name = os.path.splitext(video_filename)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        if not os.path.exists(video_output_folder):
            os.makedirs(video_output_folder)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 为每一帧命名
            frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        cap.release()
        print(f"提取了 {frame_count} 帧，保存在 {video_output_folder}")

# 使用示例

paths=[
    
    r"E:\let_go\OpenPose\GoPro\07_alde",

]





for path in paths:
    input_folder = path
    output_folder = r"E:\let_go\OpenPose\GoPro\07_alde\frame"
    extract_frames_from_videos(input_folder, output_folder)
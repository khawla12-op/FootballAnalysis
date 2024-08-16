#this will read the video and save the video using cv2
import cv2
def read_video(video_path):
    # a video capture 
    cap = cv2.VideoCapture(video_path)
    frames=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
import cv2

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("No frames to save.")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get dimensions from the first frame
    height, width, _ = output_video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()

# def save_video(ouput_video_frames,output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
#     for frame in ouput_video_frames:
#         out.write(frame)
#     out.release()
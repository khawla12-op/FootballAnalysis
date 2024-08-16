from utils import read_video ,save_video
from trackers import Tracker
def main():
    #Read Video
    video_frames =read_video("input_videos/testVideo.mp4")
    print(type(video_frames))  # Check the type here
    #Initialise the tracker
    tracker = Tracker('models/lastBest.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracks_stubs.pk1')
    #Draw output
    output_video_frames=tracker.draw_annotation(video_frames,tracks)
    #Save Video
    save_video(output_video_frames,"output_videos/output_video.avi")
if __name__=='__main__':
    main()

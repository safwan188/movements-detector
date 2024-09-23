from pytube import YouTube
from pytube import YouTube

def download_video(url, path, filename):
    try:
        yt = YouTube(url)
        # Filter to get a stream that is 1080p
        stream = yt.streams.filter(res="1080p", file_extension='mp4').first()
        if stream:
            # Specify the output file name
            stream.download(output_path=path, filename=filename)
            print("Download completed in 1080p!")
        else:
            print("1080p resolution not available, consider downloading the highest resolution available.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
url = 'https://www.youtube.com/watch?v=waTDxRZ93Qc'
download_video(url, './', 'my_video.mp4')



from moviepy.editor import VideoFileClip

def trim_video(video_path, start_time, end_time, output_path):
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    clip.write_videofile(output_path, codec='libx264')

trim_video('my_video.mp4', '00:03:18', '00:04:33', 'trimmed_video.mp4')
print("Video has been trimmed.")

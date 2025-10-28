"""
Download a single video and extract specific frames for minimal testing
"""
import argparse
import subprocess
from pathlib import Path
from pytubefix import YouTube

def download_and_extract(video_id, data_root, output_root, frame_ids):
    """Download one video and extract specific frames"""

    # Read metadata file
    txt_file = data_root / f"{video_id}.txt"
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    url = lines[0].strip()
    print(f"Downloading video from: {url}")

    # Parse timestamps - frame_ids are 0-indexed line numbers (line 0 after URL = frame 0)
    timestamps = {}
    for idx, line in enumerate(lines[1:]):
        parts = line.strip().split()
        if len(parts) > 0:
            timestamp_ms = int(parts[0])
            timestamps[idx] = timestamp_ms

    # Download video
    tmp_path = Path("temp_videos")
    tmp_path.mkdir(exist_ok=True, parents=True)

    video_file = tmp_path / f"{video_id}.mp4"

    if not video_file.exists():
        print("Downloading video...")
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        stream.download(output_path=str(tmp_path), filename=f"{video_id}.mp4")
        print("Download complete!")
    else:
        print("Video already downloaded, skipping...")

    # Create output directory
    out_path = output_root / video_id
    out_path.mkdir(exist_ok=True, parents=True)

    # Extract specific frames
    print(f"Extracting {len(frame_ids)} frames...")
    for frame_id in frame_ids:
        if frame_id in timestamps:
            timestamp_ms = timestamps[frame_id]

            # Convert timestamp from microseconds to HH:MM:SS.mmm format
            timestamp_seconds = timestamp_ms / 1000000  # microseconds to seconds
            hours = int(timestamp_seconds / 3600)
            minutes = int((timestamp_seconds % 3600) / 60)
            seconds = int(timestamp_seconds % 60)
            milliseconds = int((timestamp_seconds % 1) * 1000)

            str_timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

            # Save with timestamp as filename (as the dataset expects)
            output_file = out_path / f"{timestamp_ms}.jpg"
            if not output_file.exists():
                cmd = [
                    "ffmpeg", "-ss", str_timestamp, "-i", str(video_file),
                    "-vframes", "1", "-f", "image2", str(output_file)
                ]
                result = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result == 0:
                    print(f"  Extracted frame {frame_id} (timestamp: {timestamp_ms})")
                else:
                    print(f"  Failed to extract frame {frame_id}")
            else:
                print(f"  Frame {frame_id} already exists, skipping")

    print("Done!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, default="249fd0890d439aa9")
    parser.add_argument("--data_root", type=str, default="data/RealEstate10K/test")
    parser.add_argument("--output_root", type=str, default="data/RealEstate10K/test")
    parser.add_argument("--frames", type=int, nargs='+', default=[141, 146, 151, 81, 86, 91, 60])
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    download_and_extract(args.video_id, data_root, output_root, args.frames)

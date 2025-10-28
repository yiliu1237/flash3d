"""
Create a minimal pickle.gz file for testing with limited sequences
"""
import gzip
import pickle
import numpy as np
from pathlib import Path

def create_minimal_pickle(data_root, split, video_ids):
    """Create pickle file from .txt metadata files"""

    seq_data = {}

    for video_id in video_ids:
        txt_file = data_root / split / f"{video_id}.txt"

        if not txt_file.exists():
            print(f"Warning: {txt_file} not found, skipping...")
            continue

        with open(txt_file, 'r') as f:
            lines = f.readlines()

        url = lines[0].strip()

        # Parse camera data
        timestamps = []
        intrinsics = []
        extrinsics = []

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 19:  # timestamp + 18 camera params
                timestamp = int(parts[0])
                timestamps.append(timestamp)

                # Intrinsics: fx, fy, cx, cy
                fx = float(parts[1])
                fy = float(parts[2])
                cx = float(parts[3])
                cy = float(parts[4])
                intrinsics.append([fx, fy, cx, cy])

                # Extrinsics: rotation matrix (3x3) + translation (3x1)
                # Parts 5-13 are rotation matrix, 14-16 are translation
                rot = [float(parts[i]) for i in range(5, 14)]
                trans = [float(parts[i]) for i in range(14, 17)]
                extrinsics.append(rot + trans)

        # Combine intrinsics and extrinsics into poses
        poses = []
        for intr, extr in zip(intrinsics, extrinsics):
            # poses format: [fx, fy, cx, cy, rot(9), trans(3)]
            poses.append(intr + extr)

        seq_data[video_id] = {
            'url': url,
            'timestamps': np.array(timestamps, dtype=np.int64),
            'intrinsics': np.array(intrinsics, dtype=np.float32),
            'extrinsics': np.array(extrinsics, dtype=np.float32),
            'poses': np.array(poses, dtype=np.float32),
        }

        print(f"Processed {video_id}: {len(timestamps)} frames")

    # Save pickle file
    output_file = data_root / f"{split}.pickle.gz"
    with gzip.open(output_file, "wb") as f:
        pickle.dump(seq_data, f)

    print(f"\nCreated {output_file} with {len(seq_data)} sequences")
    return output_file

if __name__ == "__main__":
    data_root = Path("data/RealEstate10K")
    split = "test"
    video_ids = ["249fd0890d439aa9"]  # Only the video we downloaded

    create_minimal_pickle(data_root, split, video_ids)

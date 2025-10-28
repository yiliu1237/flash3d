"""
Create dummy sparse point cloud pickle for testing
"""
import gzip
import pickle
import numpy as np
from pathlib import Path

def create_dummy_sparse_pcl(output_dir, video_id):
    """Create a dummy sparse point cloud pickle file"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create minimal sparse point cloud data
    # This is just placeholder data to avoid the FileNotFoundError
    sparse_pcl_data = {
        'points': np.array([[0, 0, 1]], dtype=np.float32),  # Minimal 3D point
        'colors': np.array([[128, 128, 128]], dtype=np.uint8),  # Gray color
    }

    output_file = output_dir / f"{video_id}.pickle.gz"
    with gzip.open(output_file, "wb") as f:
        pickle.dump(sparse_pcl_data, f)

    print(f"Created dummy sparse PCL: {output_file}")
    return output_file

if __name__ == "__main__":
    # Create for our test video
    output_dir = Path("C:/Users/thero/AppData/Local/Temp/monosplat/pcl.test")
    video_id = "249fd0890d439aa9"

    create_dummy_sparse_pcl(output_dir, video_id)

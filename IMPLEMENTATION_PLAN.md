# SplatCraft Implementation Plan

**Maya Authoring Tool for 3D Gaussian Splatting**

Version: Alpha → Beta
Target: Maya 2022+ on Windows 11 / Linux
Backend: Flash3D inference engine

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Development Phases](#development-phases)
3. [Phase 0: Environment Setup](#phase-0-environment-setup)
4. [Phase 1: Flash3D Export Bridge](#phase-1-flash3d-export-bridge)
5. [Phase 2: Maya Scene Node Foundation](#phase-2-maya-scene-node-foundation)
6. [Phase 3: Viewport Proxy (VP2)](#phase-3-viewport-proxy-vp2)
7. [Phase 4: Rendered Panel (3DGS Preview)](#phase-4-rendered-panel-3dgs-preview)
8. [Phase 5: Transform Synchronization](#phase-5-transform-synchronization)
9. [Phase 6: Display Controls & LOD](#phase-6-display-controls--lod)
10. [Phase 7: Export System](#phase-7-export-system)
11. [Phase 8: UI Panel Integration](#phase-8-ui-panel-integration)
12. [Beta Features](#beta-features)
13. [Testing Strategy](#testing-strategy)
14. [Performance Targets](#performance-targets)

---

## Project Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      SplatCraft Pipeline                     │
└─────────────────────────────────────────────────────────────┘

[Input Images]
      ↓
[Flash3D Inference] (Python/PyTorch)
      ↓
[Gaussian Data Export] (.ply / .splatcraft format)
      ↓
[Maya Import] (Python API)
      ↓
┌─────────────────┐
│ SplatCraft Node │ (Single Source of Truth)
└────────┬────────┘
         │
    ┌────┴────┐
    ↓         ↓
[VP2 Proxy]  [Rendered Panel]
(Fast)       (Accurate)
    │         │
    └────┬────┘
         ↓
[Export System] (.ply / .abc / custom)
```

### Core Components

| Component | Technology | Location | Purpose |
|-----------|-----------|----------|---------|
| **Flash3D Bridge** | Python | `flash3d/export_to_maya.py` | Extract Gaussian data from inference |
| **Maya Importer** | Python (Maya API) | `maya_plugin/import_gaussians.py` | Load data into Maya scene |
| **Scene Node** | Python (MPxNode) | `maya_plugin/nodes/splatcraft_node.py` | Store Gaussian parameters |
| **VP2 Proxy** | Python/C++ (VP2 Override) | `maya_plugin/viewport/proxy_drawable.py` | Fast point cloud display |
| **Rendered Panel** | Python/OpenGL | `maya_plugin/ui/rendered_panel.py` | True 3DGS preview |
| **UI Panel** | PySide2 | `maya_plugin/ui/main_panel.py` | Control interface |
| **Export System** | Python | `maya_plugin/export_gaussians.py` | Write authored scenes |

### Data Flow

```python
# Gaussian Splat Data Structure
{
    "positions": np.ndarray,      # [N, 3] xyz coordinates
    "opacities": np.ndarray,      # [N, 1] alpha values
    "scales": np.ndarray,         # [N, 3] Gaussian radii
    "rotations": np.ndarray,      # [N, 4] quaternions
    "colors_dc": np.ndarray,      # [N, 3] RGB (SH degree 0)
    "colors_sh": np.ndarray,      # [N, sh_degree*3] optional SH
    "camera_params": {
        "intrinsics": np.ndarray, # [3, 3] K matrix
        "fov": tuple,             # (fovX, fovY)
        "transform": np.ndarray   # [4, 4] world-to-camera
    }
}
```

---

## Development Phases

### Timeline Overview

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 0** | 1-2 days | Development environment ready |
| **Phase 1** | 3-4 days | Flash3D exports Gaussian data |
| **Phase 2** | 2-3 days | Maya scene node stores data |
| **Phase 3** | 3-4 days | Point cloud proxy renders in viewport |
| **Phase 4** | 5-6 days | Rendered panel shows true 3DGS |
| **Phase 5** | 2-3 days | Transform sync between views |
| **Phase 6** | 2-3 days | LOD controls and decimation |
| **Phase 7** | 2-3 days | Export system functional |
| **Phase 8** | 2-3 days | Complete UI panel |
| **Total Alpha** | ~23-31 days | End-to-end working prototype |

---

## Phase 0: Environment Setup

**Duration:** 1-2 days
**Goal:** Prepare development environment with Flash3D and Maya

### Tasks

#### 0.1 Flash3D Environment
```bash
# Create Flash3D Python environment
cd /Users/yiliu/Desktop/SplatCraft/flash3d
conda create -y python=3.10 -n flash3d
conda activate flash3d

# Install dependencies
pip install -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Test Flash3D inference
python -m misc.download_pretrained_models -o exp/re10k_v2
```

#### 0.2 Maya Development Setup
```bash
# Create Maya plugin directory structure
mkdir -p maya_plugin/{nodes,viewport,ui,utils}
touch maya_plugin/__init__.py
touch maya_plugin/nodes/__init__.py
touch maya_plugin/viewport/__init__.py
touch maya_plugin/ui/__init__.py
touch maya_plugin/utils/__init__.py

# Install Maya Python dependencies
# (Use Maya's mayapy)
/path/to/maya/bin/mayapy -m pip install numpy PySide2
```

#### 0.3 Test Data Preparation
```bash
# Download sample images for testing
mkdir -p test_data/sample_images
mkdir -p test_data/exported_gaussians
mkdir -p test_data/maya_scenes

# Create test image set (or use provided samples)
# Recommended: 3-5 calibrated images of a simple object/room
```

### Validation Checklist
- [ ] Flash3D environment activates successfully
- [ ] Pretrained model downloads without errors
- [ ] Maya 2022+ installed and accessible
- [ ] Directory structure created
- [ ] Test images prepared

---

## Phase 1: Flash3D Export Bridge

**Duration:** 3-4 days
**Goal:** Extract Gaussian data from Flash3D inference and save to file

### 1.1 Create Export Script

**File:** `flash3d/export_to_maya.py`

```python
import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import argparse


class GaussianExporter:
    """Export Flash3D Gaussian outputs to Maya-compatible formats"""

    def __init__(self, cfg):
        self.cfg = cfg

    def extract_gaussians(self, outputs, inputs):
        """
        Extract Gaussian parameters from Flash3D outputs

        Args:
            outputs: Model output dictionary from GaussianPredictor
            inputs: Input dictionary containing camera data

        Returns:
            dict: Gaussian data ready for export
        """
        # Extract positions (gauss_means: [B, 4, N])
        positions = outputs["gauss_means"][:, :3, :].squeeze(0).T  # [N, 3]

        # Extract opacity (gauss_opacity: [B, 1, H, W])
        opacity = outputs["gauss_opacity"].squeeze(0).reshape(-1, 1)  # [N, 1]

        # Extract scaling (gauss_scaling: [B, 3, H, W])
        scales = outputs["gauss_scaling"].squeeze(0).reshape(3, -1).T  # [N, 3]

        # Extract rotation (gauss_rotation: [B, 4, H, W])
        rotations = outputs["gauss_rotation"].squeeze(0).reshape(4, -1).T  # [N, 4]

        # Extract colors (gauss_features_dc: [B, 3, H, W])
        colors_dc = outputs["gauss_features_dc"].squeeze(0).reshape(3, -1).T  # [N, 3]

        # Optional: spherical harmonics
        colors_sh = None
        if "gauss_features_rest" in outputs:
            colors_sh = outputs["gauss_features_rest"].squeeze(0)  # [sh, 3, H, W]
            sh_degree = int(np.sqrt(colors_sh.shape[0] + 1)) - 1
            colors_sh = colors_sh.reshape(-1, colors_sh.shape[-2] * colors_sh.shape[-1]).T  # [N, sh*3]

        # Extract camera parameters
        camera_params = {
            "intrinsics": inputs[("K_src", 0)].squeeze(0).cpu().numpy(),
            "inv_intrinsics": outputs[("inv_K_src", 0)].squeeze(0).cpu().numpy(),
        }

        if ("T_c2w", 0) in inputs:
            camera_params["camera_to_world"] = inputs[("T_c2w", 0)].squeeze(0).cpu().numpy()

        # Convert to numpy
        gaussian_data = {
            "positions": positions.detach().cpu().numpy(),
            "opacities": opacity.detach().cpu().numpy(),
            "scales": scales.detach().cpu().numpy(),
            "rotations": rotations.detach().cpu().numpy(),
            "colors_dc": colors_dc.detach().cpu().numpy(),
            "colors_sh": colors_sh.detach().cpu().numpy() if colors_sh is not None else None,
            "camera_params": camera_params,
            "num_gaussians": positions.shape[1]
        }

        return gaussian_data

    def save_as_ply(self, gaussian_data, output_path):
        """Save Gaussian data as PLY file"""
        positions = gaussian_data["positions"]
        scales = gaussian_data["scales"]
        rotations = gaussian_data["rotations"]
        opacities = gaussian_data["opacities"]
        colors_dc = gaussian_data["colors_dc"]

        # Convert colors from [-1, 1] to [0, 255]
        colors_uint8 = np.clip((colors_dc + 1.0) * 127.5, 0, 255).astype(np.uint8)

        # Create structured array
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # normals (unused, set to 0)
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
        ]

        elements = np.empty(positions.shape[0], dtype=dtype)
        elements['x'] = positions[:, 0]
        elements['y'] = positions[:, 1]
        elements['z'] = positions[:, 2]
        elements['nx'] = 0
        elements['ny'] = 0
        elements['nz'] = 0
        elements['red'] = colors_uint8[:, 0]
        elements['green'] = colors_uint8[:, 1]
        elements['blue'] = colors_uint8[:, 2]
        elements['scale_x'] = scales[:, 0]
        elements['scale_y'] = scales[:, 1]
        elements['scale_z'] = scales[:, 2]
        elements['rot_0'] = rotations[:, 0]
        elements['rot_1'] = rotations[:, 1]
        elements['rot_2'] = rotations[:, 2]
        elements['rot_3'] = rotations[:, 3]
        elements['opacity'] = opacities[:, 0]

        # Create PLY element
        el = PlyElement.describe(elements, 'vertex')

        # Write to file
        PlyData([el]).write(output_path)
        print(f"Saved {positions.shape[0]} Gaussians to {output_path}")

    def save_metadata(self, gaussian_data, output_path):
        """Save camera parameters and metadata as NPZ"""
        metadata_path = Path(output_path).with_suffix('.npz')

        np.savez(
            metadata_path,
            num_gaussians=gaussian_data["num_gaussians"],
            **gaussian_data["camera_params"]
        )
        print(f"Saved metadata to {metadata_path}")


def export_from_inference(cfg, model, inputs):
    """
    Run inference and export Gaussians

    Usage in evaluate.py or custom script:
        outputs = model(inputs)
        exporter = GaussianExporter(cfg)
        gaussian_data = exporter.extract_gaussians(outputs, inputs)
        exporter.save_as_ply(gaussian_data, "output.ply")
        exporter.save_metadata(gaussian_data, "output.ply")
    """
    exporter = GaussianExporter(cfg)

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)

    # Extract and save
    gaussian_data = exporter.extract_gaussians(outputs, inputs)

    return gaussian_data, exporter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Output PLY path")
    args = parser.parse_args()

    # TODO: Load config, model, run inference, and export
    print(f"Export script ready. Implement full pipeline in evaluate.py or custom script.")
```

### 1.2 Integration Script

**File:** `flash3d/inference_and_export.py`

```python
"""
Standalone script to run Flash3D inference and export for Maya
"""
import os
import sys
import torch
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torchvision.transforms as transforms

# Add flash3d to path
sys.path.insert(0, str(Path(__file__).parent))

from models.model import GaussianPredictor
from export_to_maya import GaussianExporter


def load_image(image_path, height=384, width=512):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run inference and export"""

    # Override config for single image inference
    cfg.data_loader.batch_size = 1
    cfg.model.gaussians_per_pixel = cfg.model.get('gaussians_per_pixel', 1)
    cfg.dataset.height = 384
    cfg.dataset.width = 512
    cfg.dataset.pad_border_aug = 0

    # Load model
    print("Loading model...")
    model = GaussianPredictor(cfg)
    model.load_model(cfg.checkpoint_path, device="cuda")
    model.set_eval()
    model = model.cuda()

    # Load image
    print(f"Loading image: {cfg.image_path}")
    img_tensor = load_image(cfg.image_path, cfg.dataset.height, cfg.dataset.width).cuda()

    # Prepare inputs
    inputs = {
        ("color_aug", 0, 0): img_tensor,
        "target_frame_ids": [],
    }

    # Optional: add intrinsics if known
    if hasattr(cfg, 'intrinsics'):
        K = torch.tensor(cfg.intrinsics, dtype=torch.float32).unsqueeze(0).cuda()
        inputs[("K_src", 0)] = K

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs)

    # Export
    print("Exporting Gaussians...")
    exporter = GaussianExporter(cfg)
    gaussian_data = exporter.extract_gaussians(outputs, inputs)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exporter.save_as_ply(gaussian_data, str(output_path))
    exporter.save_metadata(gaussian_data, str(output_path))

    print(f"\n✓ Export complete!")
    print(f"  Gaussians: {gaussian_data['num_gaussians']}")
    print(f"  PLY: {output_path}")
    print(f"  Metadata: {output_path.with_suffix('.npz')}")


if __name__ == "__main__":
    # Example usage:
    # python inference_and_export.py \
    #   checkpoint_path=exp/re10k_v2 \
    #   image_path=test_data/sample.jpg \
    #   output_path=test_data/exported_gaussians/sample.ply
    main()
```

### 1.3 Testing

```bash
# Test export with sample image
cd flash3d
conda activate flash3d

python inference_and_export.py \
  checkpoint_path=exp/re10k_v2 \
  image_path=../test_data/sample_images/test01.jpg \
  output_path=../test_data/exported_gaussians/test01.ply
```

### Validation Checklist
- [ ] Export script creates .ply file
- [ ] Metadata .npz file created
- [ ] Gaussian count matches expected range
- [ ] PLY file opens in MeshLab/CloudCompare
- [ ] Colors visible in point cloud viewer

---

## Phase 2: Maya Scene Node Foundation

**Duration:** 2-3 days
**Goal:** Create custom Maya node to store Gaussian data

### 2.1 Scene Node Implementation

**File:** `maya_plugin/nodes/splatcraft_node.py`

```python
import sys
import maya.api.OpenMaya as om
import maya.api.OpenMayaRender as omr
import numpy as np


def maya_useNewAPI():
    """Tell Maya to use Maya Python API 2.0"""
    pass


class SplatCraftNode(om.MPxNode):
    """
    Custom Maya node to store 3D Gaussian Splatting data

    Attributes:
        - gaussianData: Binary blob containing all Gaussian parameters
        - numGaussians: Number of Gaussians in the scene
        - displayLOD: Level of detail for viewport display (0.0-1.0)
        - pointSize: Display point size
        - enableRender: Toggle rendered panel display
    """

    TYPE_NAME = "splatCraftNode"
    TYPE_ID = om.MTypeId(0x00138200)  # Unique ID - register with Autodesk if publishing

    # Attributes
    gaussian_data_attr = None
    num_gaussians_attr = None
    display_lod_attr = None
    point_size_attr = None
    enable_render_attr = None

    # Gaussian parameter arrays (cached)
    positions = None
    opacities = None
    scales = None
    rotations = None
    colors_dc = None
    colors_sh = None

    def __init__(self):
        om.MPxNode.__init__(self)

    @classmethod
    def creator(cls):
        return cls()

    @classmethod
    def initialize(cls):
        """Define node attributes"""

        # Typed attribute for binary data
        typed_attr = om.MFnTypedAttribute()
        cls.gaussian_data_attr = typed_attr.create(
            "gaussianData", "gd", om.MFnData.kString
        )
        typed_attr.writable = True
        typed_attr.storable = True
        cls.addAttribute(cls.gaussian_data_attr)

        # Numeric attributes
        num_attr = om.MFnNumericAttribute()

        cls.num_gaussians_attr = num_attr.create(
            "numGaussians", "ng", om.MFnNumericData.kInt, 0
        )
        num_attr.writable = True
        num_attr.storable = True
        num_attr.readable = True
        cls.addAttribute(cls.num_gaussians_attr)

        cls.display_lod_attr = num_attr.create(
            "displayLOD", "lod", om.MFnNumericData.kFloat, 1.0
        )
        num_attr.writable = True
        num_attr.storable = True
        num_attr.setMin(0.0)
        num_attr.setMax(1.0)
        cls.addAttribute(cls.display_lod_attr)

        cls.point_size_attr = num_attr.create(
            "pointSize", "ps", om.MFnNumericData.kFloat, 2.0
        )
        num_attr.writable = True
        num_attr.storable = True
        num_attr.setMin(0.1)
        num_attr.setMax(20.0)
        cls.addAttribute(cls.point_size_attr)

        cls.enable_render_attr = num_attr.create(
            "enableRender", "er", om.MFnNumericData.kBoolean, True
        )
        num_attr.writable = True
        num_attr.storable = True
        cls.addAttribute(cls.enable_render_attr)

    def compute(self, plug, data_block):
        """Compute method (not used for data storage node)"""
        return om.kUnknownParameter

    def set_gaussian_data(self, gaussian_dict):
        """
        Store Gaussian data in node

        Args:
            gaussian_dict: Dictionary with keys:
                - positions: [N, 3]
                - opacities: [N, 1]
                - scales: [N, 3]
                - rotations: [N, 4]
                - colors_dc: [N, 3]
                - colors_sh: [N, sh*3] (optional)
        """
        # Cache numpy arrays
        self.positions = gaussian_dict["positions"]
        self.opacities = gaussian_dict["opacities"]
        self.scales = gaussian_dict["scales"]
        self.rotations = gaussian_dict["rotations"]
        self.colors_dc = gaussian_dict["colors_dc"]
        self.colors_sh = gaussian_dict.get("colors_sh", None)

        # Update num_gaussians attribute
        plug = om.MPlug(self.thisMObject(), self.num_gaussians_attr)
        plug.setInt(self.positions.shape[0])

    def get_gaussian_data(self):
        """Retrieve Gaussian data from node"""
        return {
            "positions": self.positions,
            "opacities": self.opacities,
            "scales": self.scales,
            "rotations": self.rotations,
            "colors_dc": self.colors_dc,
            "colors_sh": self.colors_sh,
        }

    def get_decimated_points(self, lod_factor=1.0):
        """
        Get decimated point cloud for viewport display

        Args:
            lod_factor: 0.0 to 1.0, percentage of points to show

        Returns:
            tuple: (positions, colors) as numpy arrays
        """
        if self.positions is None:
            return None, None

        num_points = int(self.positions.shape[0] * lod_factor)
        if num_points == 0:
            return None, None

        # Random sampling
        indices = np.random.choice(self.positions.shape[0], num_points, replace=False)

        positions = self.positions[indices]
        colors = np.clip((self.colors_dc[indices] + 1.0) * 0.5, 0, 1)  # [-1,1] -> [0,1]

        return positions, colors


def initializePlugin(plugin):
    """Initialize the plugin"""
    vendor = "SplatCraft"
    version = "0.1.0"

    plugin_fn = om.MFnPlugin(plugin, vendor, version)

    try:
        plugin_fn.registerNode(
            SplatCraftNode.TYPE_NAME,
            SplatCraftNode.TYPE_ID,
            SplatCraftNode.creator,
            SplatCraftNode.initialize,
            om.MPxNode.kDependNode
        )
    except:
        om.MGlobal.displayError(f"Failed to register node: {SplatCraftNode.TYPE_NAME}")
        raise


def uninitializePlugin(plugin):
    """Uninitialize the plugin"""
    plugin_fn = om.MFnPlugin(plugin)

    try:
        plugin_fn.deregisterNode(SplatCraftNode.TYPE_ID)
    except:
        om.MGlobal.displayError(f"Failed to deregister node: {SplatCraftNode.TYPE_NAME}")
        raise
```

### 2.2 Import Utility

**File:** `maya_plugin/import_gaussians.py`

```python
import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om
from plyfile import PlyData
from pathlib import Path


def read_ply_gaussians(ply_path):
    """
    Read Gaussian splat data from PLY file

    Returns:
        dict: Gaussian parameters
    """
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']

    # Extract positions
    positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

    # Extract colors (convert from uint8 to [-1, 1])
    colors_uint8 = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1)
    colors_dc = (colors_uint8.astype(np.float32) / 127.5) - 1.0

    # Extract scales
    scales = np.stack([vertex['scale_x'], vertex['scale_y'], vertex['scale_z']], axis=1)

    # Extract rotations
    rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)

    # Extract opacity
    opacities = vertex['opacity'].reshape(-1, 1)

    return {
        "positions": positions.astype(np.float32),
        "opacities": opacities.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "colors_dc": colors_dc.astype(np.float32),
        "colors_sh": None,
    }


def read_metadata(npz_path):
    """Read camera metadata from NPZ file"""
    if not Path(npz_path).exists():
        return None

    metadata = np.load(npz_path)
    return {k: metadata[k] for k in metadata.files}


def import_gaussian_scene(ply_path):
    """
    Import Gaussian splat scene into Maya

    Args:
        ply_path: Path to .ply file

    Returns:
        str: Name of created SplatCraft node
    """
    # Read Gaussian data
    print(f"Reading Gaussians from {ply_path}...")
    gaussian_data = read_ply_gaussians(ply_path)

    # Read metadata
    npz_path = Path(ply_path).with_suffix('.npz')
    metadata = read_metadata(npz_path)

    # Create SplatCraft node
    node_name = cmds.createNode("splatCraftNode", name="splatCraftNode1")
    print(f"Created node: {node_name}")

    # Get MObject for the node
    sel_list = om.MSelectionList()
    sel_list.add(node_name)
    node_obj = sel_list.getDependNode(0)

    # Get the node function set
    dep_fn = om.MFnDependencyNode(node_obj)

    # HACK: Access the MPxNode instance directly to call custom methods
    # In production, use commands or better API patterns
    # For now, store data as scriptJob or use custom command

    # Create transform node
    transform_name = cmds.createNode("transform", name="splatCraftTransform1")
    cmds.parent(node_name, transform_name, add=True, shape=True)

    # Set initial attributes
    cmds.setAttr(f"{node_name}.numGaussians", gaussian_data["positions"].shape[0])
    cmds.setAttr(f"{node_name}.displayLOD", 0.1)  # Start with 10% for performance
    cmds.setAttr(f"{node_name}.pointSize", 2.0)

    print(f"✓ Imported {gaussian_data['positions'].shape[0]} Gaussians")
    print(f"  Transform: {transform_name}")
    print(f"  Node: {node_name}")

    # TODO: Actually store the data in the node (requires command or better pattern)
    # For now, return the node name for manual data assignment

    return node_name, gaussian_data


# Example usage in Maya Script Editor:
# import sys
# sys.path.append('/Users/yiliu/Desktop/SplatCraft/maya_plugin')
# import import_gaussians
# import_gaussians.import_gaussian_scene('/Users/yiliu/Desktop/SplatCraft/test_data/exported_gaussians/test01.ply')
```

### 2.3 Testing in Maya

```python
# In Maya Script Editor (Python):

import sys
sys.path.append('/Users/yiliu/Desktop/SplatCraft/maya_plugin')

# Load plugin
import maya.cmds as cmds
cmds.loadPlugin('/Users/yiliu/Desktop/SplatCraft/maya_plugin/nodes/splatcraft_node.py')

# Import Gaussian scene
import import_gaussians
node_name, data = import_gaussians.import_gaussian_scene(
    '/Users/yiliu/Desktop/SplatCraft/test_data/exported_gaussians/test01.ply'
)

print(f"Loaded {data['positions'].shape[0]} Gaussians into {node_name}")
```

### Validation Checklist
- [ ] Plugin loads in Maya without errors
- [ ] SplatCraft node appears in Node Editor
- [ ] Import script reads PLY file correctly
- [ ] Node stores Gaussian count attribute
- [ ] Transform node created and linked

---

## Phase 3: Viewport Proxy (VP2)

**Duration:** 3-4 days
**Goal:** Display point cloud proxy in Maya viewport

### 3.1 Viewport 2.0 Override (Python)

**File:** `maya_plugin/viewport/proxy_drawable.py`

```python
import maya.api.OpenMaya as om
import maya.api.OpenMayaRender as omr
import numpy as np


class SplatCraftDrawOverride(omr.MPxDrawOverride):
    """
    Viewport 2.0 draw override for SplatCraft node
    Renders decimated point cloud for fast interaction
    """

    NAME = "SplatCraftDrawOverride"

    def __init__(self, obj):
        omr.MPxDrawOverride.__init__(self, obj, None, False)
        self.positions = None
        self.colors = None

    @staticmethod
    def creator(obj):
        return SplatCraftDrawOverride(obj)

    @staticmethod
    def draw(context, data):
        return

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices

    def prepareForDraw(self, obj_path, camera_path, frame_context, old_data):
        """
        Prepare point cloud data for drawing
        Called before draw()
        """
        # Get the SplatCraft node
        node = obj_path.node()

        # TODO: Retrieve decimated points from node
        # This requires accessing the custom node's Python instance
        # For now, create dummy data

        # Placeholder: create random points
        if self.positions is None:
            num_points = 1000
            self.positions = np.random.randn(num_points, 3).astype(np.float32)
            self.colors = np.random.rand(num_points, 3).astype(np.float32)

        return None

    def hasUIDrawables(self):
        return True

    def addUIDrawables(self, obj_path, draw_manager, frame_context, data):
        """
        Draw point cloud in viewport
        """
        if self.positions is None:
            return

        draw_manager.beginDrawable()

        # Set point size
        draw_manager.setPointSize(2.0)

        # Draw points
        point_list = om.MPointArray()
        color_list = om.MColorArray()

        for i in range(len(self.positions)):
            point_list.append(om.MPoint(
                self.positions[i, 0],
                self.positions[i, 1],
                self.positions[i, 2]
            ))
            color_list.append(om.MColor([
                self.colors[i, 0],
                self.colors[i, 1],
                self.colors[i, 2],
                1.0
            ]))

        draw_manager.setColorArray(color_list)
        draw_manager.mesh(omr.MUIDrawManager.kPoints, point_list)

        draw_manager.endDrawable()


def register_draw_override():
    """Register the draw override"""
    try:
        omr.MDrawRegistry.registerDrawOverrideCreator(
            "drawdb/geometry/splatCraftNode",
            "splatCraftNode",
            SplatCraftDrawOverride.creator
        )
        print("✓ Registered SplatCraft draw override")
    except:
        om.MGlobal.displayError("Failed to register draw override")
        raise


def deregister_draw_override():
    """Deregister the draw override"""
    try:
        omr.MDrawRegistry.deregisterDrawOverrideCreator(
            "drawdb/geometry/splatCraftNode",
            "splatCraftNode"
        )
        print("✓ Deregistered SplatCraft draw override")
    except:
        om.MGlobal.displayError("Failed to deregister draw override")
        raise
```

### 3.2 Update Plugin Initialization

**File:** `maya_plugin/nodes/splatcraft_node.py` (append to existing file)

```python
# Add to initializePlugin():
def initializePlugin(plugin):
    """Initialize the plugin"""
    vendor = "SplatCraft"
    version = "0.1.0"

    plugin_fn = om.MFnPlugin(plugin, vendor, version)

    try:
        # Register node
        plugin_fn.registerNode(
            SplatCraftNode.TYPE_NAME,
            SplatCraftNode.TYPE_ID,
            SplatCraftNode.creator,
            SplatCraftNode.initialize,
            om.MPxNode.kLocatorNode  # Changed from kDependNode to kLocatorNode
        )

        # Register draw override
        from viewport.proxy_drawable import register_draw_override
        register_draw_override()

    except:
        om.MGlobal.displayError(f"Failed to register node: {SplatCraftNode.TYPE_NAME}")
        raise


def uninitializePlugin(plugin):
    """Uninitialize the plugin"""
    plugin_fn = om.MFnPlugin(plugin)

    try:
        # Deregister draw override
        from viewport.proxy_drawable import deregister_draw_override
        deregister_draw_override()

        # Deregister node
        plugin_fn.deregisterNode(SplatCraftNode.TYPE_ID)
    except:
        om.MGlobal.displayError(f"Failed to deregister node: {SplatCraftNode.TYPE_NAME}")
        raise
```

### 3.3 Testing

```python
# In Maya Script Editor:
import maya.cmds as cmds

# Reload plugin
cmds.unloadPlugin('splatcraft_node.py', force=True)
cmds.loadPlugin('/Users/yiliu/Desktop/SplatCraft/maya_plugin/nodes/splatcraft_node.py')

# Create node
node = cmds.createNode('splatCraftNode')
cmds.select(node)

# Check viewport - should see point cloud
```

### Validation Checklist
- [ ] Point cloud visible in viewport
- [ ] Points have correct colors
- [ ] Can select and transform node
- [ ] Performance acceptable (>30 FPS for 10k points)
- [ ] LOD slider reduces point count

---

## Phase 4: Rendered Panel (3DGS Preview)

**Duration:** 5-6 days
**Goal:** Display true 3DGS rendering in separate panel

**Note:** This is the most complex phase. We'll start with a simplified OpenGL renderer.

### 4.1 Simplified Splat Renderer

**File:** `maya_plugin/rendering/splat_renderer.py`

```python
"""
Simplified 3D Gaussian Splatting renderer using OpenGL
For production, consider porting diff-gaussian-rasterization
"""
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes


class SplatRenderer:
    """
    Simple point-based splat renderer
    Later upgrade to true Gaussian rasterization
    """

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.vbo = None
        self.vao = None
        self.shader_program = None

    def initialize(self):
        """Initialize OpenGL resources"""
        self.create_shaders()
        self.create_buffers()

    def create_shaders(self):
        """Create vertex and fragment shaders"""

        # Vertex shader - simple point splatting
        vertex_shader_code = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 color;
        layout (location = 2) in float opacity;
        layout (location = 3) in float scale;

        uniform mat4 modelViewProjection;
        uniform float pointSize;

        out vec3 fragColor;
        out float fragOpacity;

        void main() {
            gl_Position = modelViewProjection * vec4(position, 1.0);
            gl_PointSize = pointSize * scale / gl_Position.w;  // Perspective scaling
            fragColor = color;
            fragOpacity = opacity;
        }
        """

        # Fragment shader - circular splat with alpha
        fragment_shader_code = """
        #version 330 core
        in vec3 fragColor;
        in float fragOpacity;

        out vec4 FragColor;

        void main() {
            // Create circular splat
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            if (dist > 0.5) discard;

            // Gaussian falloff
            float alpha = fragOpacity * exp(-dist * dist * 8.0);
            FragColor = vec4(fragColor, alpha);
        }
        """

        # Compile shaders
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_shader_code)
        glCompileShader(vertex_shader)

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_shader_code)
        glCompileShader(fragment_shader)

        # Link program
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def create_buffers(self):
        """Create VBO and VAO"""
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

    def upload_data(self, positions, colors, opacities, scales):
        """Upload Gaussian data to GPU"""
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Interleave data: [x, y, z, r, g, b, opacity, scale]
        num_points = positions.shape[0]
        vertex_data = np.zeros((num_points, 8), dtype=np.float32)
        vertex_data[:, 0:3] = positions
        vertex_data[:, 3:6] = np.clip((colors + 1.0) * 0.5, 0, 1)  # [-1,1] -> [0,1]
        vertex_data[:, 6] = opacities.squeeze()
        vertex_data[:, 7] = np.mean(scales, axis=1)  # Average scale for simplicity

        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Opacity attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        # Scale attribute
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(28))
        glEnableVertexAttribArray(3)

        self.num_points = num_points

    def render(self, mvp_matrix, point_size=10.0):
        """
        Render Gaussians

        Args:
            mvp_matrix: 4x4 model-view-projection matrix
            point_size: Base point size
        """
        glUseProgram(self.shader_program)

        # Set uniforms
        mvp_loc = glGetUniformLocation(self.shader_program, "modelViewProjection")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_matrix.astype(np.float32))

        ps_loc = glGetUniformLocation(self.shader_program, "pointSize")
        glUniform1f(ps_loc, point_size)

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)

        # Draw
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)

        glDisable(GL_BLEND)

    def cleanup(self):
        """Release OpenGL resources"""
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
```

### 4.2 Rendered Panel Widget

**File:** `maya_plugin/ui/rendered_panel.py`

```python
"""
PySide2 widget for rendered 3DGS panel
"""
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtOpenGL import QGLWidget
from OpenGL.GL import *
import numpy as np

try:
    from rendering.splat_renderer import SplatRenderer
except:
    print("Warning: Could not import SplatRenderer")


class RenderedPanel(QGLWidget):
    """OpenGL widget to display true 3DGS rendering"""

    def __init__(self, parent=None):
        super(RenderedPanel, self).__init__(parent)

        self.renderer = None
        self.gaussian_data = None
        self.camera_distance = 5.0
        self.camera_rotation = [0, 0]
        self.last_mouse_pos = None

        # Set minimum size
        self.setMinimumSize(400, 300)

    def initializeGL(self):
        """Initialize OpenGL context"""
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Create renderer
        self.renderer = SplatRenderer(self.width(), self.height())
        self.renderer.initialize()

    def resizeGL(self, width, height):
        """Handle widget resize"""
        glViewport(0, 0, width, height)

    def paintGL(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.gaussian_data is None or self.renderer is None:
            return

        # Build MVP matrix
        mvp = self.get_mvp_matrix()

        # Render
        self.renderer.render(mvp, point_size=20.0)

    def get_mvp_matrix(self):
        """Get model-view-projection matrix"""
        # Projection matrix
        fov = 45.0
        aspect = self.width() / float(self.height())
        near = 0.1
        far = 100.0

        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        # View matrix (simple orbit camera)
        rx = self.camera_rotation[1]
        ry = self.camera_rotation[0]

        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -self.camera_distance  # Move back

        # Rotation
        rot_x = np.eye(4, dtype=np.float32)
        rot_x[1, 1] = np.cos(np.radians(rx))
        rot_x[1, 2] = -np.sin(np.radians(rx))
        rot_x[2, 1] = np.sin(np.radians(rx))
        rot_x[2, 2] = np.cos(np.radians(rx))

        rot_y = np.eye(4, dtype=np.float32)
        rot_y[0, 0] = np.cos(np.radians(ry))
        rot_y[0, 2] = np.sin(np.radians(ry))
        rot_y[2, 0] = -np.sin(np.radians(ry))
        rot_y[2, 2] = np.cos(np.radians(ry))

        view = view @ rot_x @ rot_y

        # Model matrix (identity for now)
        model = np.eye(4, dtype=np.float32)

        mvp = projection @ view @ model
        return mvp

    def set_gaussian_data(self, gaussian_data):
        """
        Load Gaussian data into renderer

        Args:
            gaussian_data: Dict with positions, colors, opacities, scales
        """
        self.gaussian_data = gaussian_data

        if self.renderer:
            self.renderer.upload_data(
                gaussian_data["positions"],
                gaussian_data["colors_dc"],
                gaussian_data["opacities"],
                gaussian_data["scales"]
            )
            self.update()

    def mousePressEvent(self, event):
        """Handle mouse press"""
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera rotation"""
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            self.camera_rotation[0] += dx * 0.5
            self.camera_rotation[1] += dy * 0.5

            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        self.camera_distance *= 1.0 - (delta / 1200.0)
        self.camera_distance = np.clip(self.camera_distance, 0.5, 50.0)
        self.update()
```

### 4.3 Testing

```python
# Test rendered panel standalone
from PySide2 import QtWidgets
import sys
sys.path.append('/Users/yiliu/Desktop/SplatCraft/maya_plugin')

from ui.rendered_panel import RenderedPanel
from import_gaussians import read_ply_gaussians

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication(sys.argv)

# Create panel
panel = RenderedPanel()
panel.setWindowTitle("SplatCraft Rendered Panel")

# Load data
gaussian_data = read_ply_gaussians('/Users/yiliu/Desktop/SplatCraft/test_data/exported_gaussians/test01.ply')
panel.set_gaussian_data(gaussian_data)

panel.show()
app.exec_()
```

### Validation Checklist
- [ ] Rendered panel opens without errors
- [ ] Gaussians visible as colored points
- [ ] Mouse drag rotates camera
- [ ] Mouse wheel zooms
- [ ] Alpha blending works correctly

---

## Phase 5: Transform Synchronization

**Duration:** 2-3 days
**Goal:** Keep viewport proxy and rendered panel synchronized

### 5.1 Transform Monitor

**File:** `maya_plugin/utils/transform_monitor.py`

```python
"""
Monitor Maya transform changes and sync to rendered panel
"""
import maya.api.OpenMaya as om
import maya.cmds as cmds


class TransformMonitor:
    """
    Monitors transform changes on SplatCraft nodes
    and notifies rendered panel for updates
    """

    def __init__(self, node_name, rendered_panel):
        self.node_name = node_name
        self.rendered_panel = rendered_panel
        self.callback_id = None

        # Get transform node
        self.transform_name = cmds.listRelatives(node_name, parent=True, type='transform')[0]

    def start_monitoring(self):
        """Start monitoring transform changes"""
        # Get MObject for transform
        sel_list = om.MSelectionList()
        sel_list.add(self.transform_name)
        transform_obj = sel_list.getDependNode(0)

        # Register callback
        self.callback_id = om.MNodeMessage.addAttributeChangedCallback(
            transform_obj,
            self.on_transform_changed
        )

        print(f"Started monitoring transforms for {self.transform_name}")

    def stop_monitoring(self):
        """Stop monitoring"""
        if self.callback_id:
            om.MMessage.removeCallback(self.callback_id)
            self.callback_id = None

    def on_transform_changed(self, msg, plug, other_plug, client_data):
        """Callback when transform changes"""
        if msg & om.MNodeMessage.kAttributeSet:
            # Check if it's a transform attribute
            attr_name = plug.partialName()
            if any(t in attr_name for t in ['translate', 'rotate', 'scale']):
                self.update_rendered_panel()

    def update_rendered_panel(self):
        """Update the rendered panel with new transform"""
        # Get world matrix
        world_matrix = cmds.xform(self.transform_name, query=True, matrix=True, worldSpace=True)

        # Update rendered panel
        if self.rendered_panel:
            self.rendered_panel.set_transform(world_matrix)

    def get_world_matrix(self):
        """Get current world transform matrix"""
        return cmds.xform(self.transform_name, query=True, matrix=True, worldSpace=True)
```

### 5.2 Update Rendered Panel

**File:** `maya_plugin/ui/rendered_panel.py` (add method)

```python
# Add to RenderedPanel class:

def set_transform(self, world_matrix):
    """
    Update model transform from Maya

    Args:
        world_matrix: 16-element list representing 4x4 matrix (row-major)
    """
    # Convert Maya matrix (row-major) to numpy (column-major)
    self.model_matrix = np.array(world_matrix, dtype=np.float32).reshape(4, 4).T
    self.update()

def get_mvp_matrix(self):
    """Get model-view-projection matrix"""
    # ... (existing projection and view code) ...

    # Use Maya transform as model matrix
    model = getattr(self, 'model_matrix', np.eye(4, dtype=np.float32))

    mvp = projection @ view @ model
    return mvp
```

### Validation Checklist
- [ ] Moving viewport object updates rendered panel
- [ ] Rotation synchronized
- [ ] Scale synchronized
- [ ] No lag or flicker during transform

---

## Phase 6: Display Controls & LOD

**Duration:** 2-3 days
**Goal:** Add LOD slider and display controls

### 6.1 Control Panel

**File:** `maya_plugin/ui/control_panel.py`

```python
"""
Main SplatCraft UI control panel
"""
from PySide2 import QtWidgets, QtCore


class SplatCraftControlPanel(QtWidgets.QWidget):
    """Main control panel for SplatCraft"""

    # Signals
    lod_changed = QtCore.Signal(float)
    point_size_changed = QtCore.Signal(float)
    render_enabled_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(SplatCraftControlPanel, self).__init__(parent)

        self.setup_ui()

    def setup_ui(self):
        """Create UI layout"""
        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("<h2>SplatCraft</h2>")
        layout.addWidget(title)

        # Import section
        import_group = QtWidgets.QGroupBox("Import")
        import_layout = QtWidgets.QVBoxLayout()

        self.import_btn = QtWidgets.QPushButton("Import Gaussian PLY...")
        self.import_btn.clicked.connect(self.on_import_clicked)
        import_layout.addWidget(self.import_btn)

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        # Display controls
        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QFormLayout()

        # LOD slider
        self.lod_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lod_slider.setMinimum(1)
        self.lod_slider.setMaximum(100)
        self.lod_slider.setValue(10)
        self.lod_slider.valueChanged.connect(self.on_lod_changed)

        self.lod_label = QtWidgets.QLabel("10%")
        lod_layout = QtWidgets.QHBoxLayout()
        lod_layout.addWidget(self.lod_slider)
        lod_layout.addWidget(self.lod_label)

        display_layout.addRow("LOD:", lod_layout)

        # Point size
        self.point_size_spin = QtWidgets.QDoubleSpinBox()
        self.point_size_spin.setMinimum(0.1)
        self.point_size_spin.setMaximum(20.0)
        self.point_size_spin.setValue(2.0)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)

        display_layout.addRow("Point Size:", self.point_size_spin)

        # Render toggle
        self.render_checkbox = QtWidgets.QCheckBox("Enable Rendered Panel")
        self.render_checkbox.setChecked(True)
        self.render_checkbox.stateChanged.connect(self.on_render_toggled)

        display_layout.addRow(self.render_checkbox)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Export section
        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout()

        self.export_btn = QtWidgets.QPushButton("Export Gaussian PLY...")
        self.export_btn.clicked.connect(self.on_export_clicked)
        export_layout.addWidget(self.export_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Status
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def on_import_clicked(self):
        """Handle import button click"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Gaussian PLY",
            "",
            "PLY Files (*.ply)"
        )

        if file_path:
            self.status_label.setText(f"Importing {file_path}...")
            # TODO: Call import function
            print(f"Import: {file_path}")

    def on_export_clicked(self):
        """Handle export button click"""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Gaussian PLY",
            "",
            "PLY Files (*.ply)"
        )

        if file_path:
            self.status_label.setText(f"Exporting to {file_path}...")
            # TODO: Call export function
            print(f"Export: {file_path}")

    def on_lod_changed(self, value):
        """Handle LOD slider change"""
        lod_factor = value / 100.0
        self.lod_label.setText(f"{value}%")
        self.lod_changed.emit(lod_factor)

    def on_point_size_changed(self, value):
        """Handle point size change"""
        self.point_size_changed.emit(value)

    def on_render_toggled(self, state):
        """Handle render checkbox toggle"""
        enabled = state == QtCore.Qt.Checked
        self.render_enabled_changed.emit(enabled)
```

### Validation Checklist
- [ ] Control panel displays correctly
- [ ] LOD slider changes viewport density
- [ ] Point size affects both views
- [ ] Import/export dialogs open
- [ ] Status updates correctly

---

## Phase 7: Export System

**Duration:** 2-3 days
**Goal:** Export authored Gaussian scenes

### 7.1 Export Implementation

**File:** `maya_plugin/export_gaussians.py`

```python
"""
Export authored Gaussian scenes from Maya
"""
import numpy as np
import maya.cmds as cmds
from plyfile import PlyData, PlyElement
from pathlib import Path


def export_gaussian_scene(node_name, output_path, include_transform=True):
    """
    Export SplatCraft node to PLY file

    Args:
        node_name: Name of SplatCraft node
        output_path: Path to output .ply file
        include_transform: Apply Maya transform to positions
    """
    # Get Gaussian data from node
    # TODO: Access node's Python instance to get data
    # For now, placeholder

    # Get transform if requested
    if include_transform:
        transform_name = cmds.listRelatives(node_name, parent=True, type='transform')[0]
        world_matrix = np.array(
            cmds.xform(transform_name, query=True, matrix=True, worldSpace=True)
        ).reshape(4, 4).T  # Convert to column-major

    # Apply transform to positions
    # positions_transformed = apply_transform(positions, world_matrix)

    # Save PLY
    # save_ply(output_path, positions_transformed, colors, opacities, scales, rotations)

    print(f"✓ Exported to {output_path}")


def apply_transform(positions, matrix):
    """Apply 4x4 transform matrix to positions"""
    # Convert to homogeneous coordinates
    ones = np.ones((positions.shape[0], 1))
    positions_homo = np.hstack([positions, ones])

    # Apply transform
    transformed = (matrix @ positions_homo.T).T

    # Convert back to 3D
    return transformed[:, :3] / transformed[:, 3:4]
```

### Validation Checklist
- [ ] Export creates valid PLY file
- [ ] Transform applied correctly
- [ ] Re-import matches original
- [ ] Metadata preserved

---

## Phase 8: UI Panel Integration

**Duration:** 2-3 days
**Goal:** Integrate all components into Maya UI

### 8.1 Main Panel

**File:** `maya_plugin/ui/main_panel.py`

```python
"""
Main SplatCraft panel for Maya
"""
from PySide2 import QtWidgets
import maya.cmds as cmds
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin


class SplatCraftMainPanel(MayaQWidgetDockableMixin, QtWidgets.QWidget):
    """Dockable Maya panel for SplatCraft"""

    PANEL_NAME = "SplatCraftPanel"

    def __init__(self, parent=None):
        super(SplatCraftMainPanel, self).__init__(parent=parent)

        self.setWindowTitle("SplatCraft")
        self.setObjectName(self.PANEL_NAME)

        self.setup_ui()

    def setup_ui(self):
        """Create UI"""
        layout = QtWidgets.QVBoxLayout(self)

        # Control panel
        from ui.control_panel import SplatCraftControlPanel
        self.control_panel = SplatCraftControlPanel()
        layout.addWidget(self.control_panel)

        # Rendered panel
        from ui.rendered_panel import RenderedPanel
        self.rendered_panel = RenderedPanel()
        layout.addWidget(self.rendered_panel, stretch=1)

        # Connect signals
        self.control_panel.lod_changed.connect(self.on_lod_changed)
        self.control_panel.point_size_changed.connect(self.on_point_size_changed)

    def on_lod_changed(self, lod_factor):
        """Handle LOD change"""
        # Update viewport proxy
        # Update rendered panel
        print(f"LOD changed: {lod_factor}")

    def on_point_size_changed(self, point_size):
        """Handle point size change"""
        print(f"Point size changed: {point_size}")


def show_panel():
    """Show or create SplatCraft panel"""
    # Delete existing
    if cmds.workspaceControl(SplatCraftMainPanel.PANEL_NAME, exists=True):
        cmds.deleteUI(SplatCraftMainPanel.PANEL_NAME)

    # Create new
    panel = SplatCraftMainPanel()
    panel.show(dockable=True, area='right', allowedArea='all')

    return panel


# Usage in Maya:
# import sys
# sys.path.append('/Users/yiliu/Desktop/SplatCraft/maya_plugin')
# from ui.main_panel import show_panel
# show_panel()
```

### Validation Checklist
- [ ] Panel docks in Maya UI
- [ ] All controls functional
- [ ] Rendered panel integrated
- [ ] Panel persists across Maya sessions

---

## Beta Features

### B1: Multi-Capture Batch Import
- Import multiple PLY files
- Automatic scene assembly
- Alignment tools

### B2: Cluster/Group Transforms
- Select subsets of Gaussians
- Group transforms
- Hierarchical editing

### B3: Export Presets
- Multiple export formats
- Compression options
- Metadata preservation

### B4: Error Handling & Progress
- Progress bars for long operations
- Error dialogs
- Cancel/resume support

---

## Testing Strategy

### Unit Tests
- Gaussian data I/O
- Transform math
- LOD decimation

### Integration Tests
- Full pipeline (image → inference → import → export)
- Transform synchronization
- Multi-scene handling

### Performance Tests
- 10k, 100k, 1M Gaussian benchmarks
- Frame rate targets (>30 FPS viewport, >15 FPS rendered)
- Memory profiling

### User Acceptance Tests
- End-to-end workflow tests
- Usability evaluation
- Bug tracking

---

## Performance Targets

| Metric | Alpha Target | Beta Target |
|--------|-------------|-------------|
| **Viewport FPS** (100k points) | >30 FPS | >60 FPS |
| **Rendered FPS** (100k points) | >15 FPS | >30 FPS |
| **Import time** (100k points) | <5 sec | <2 sec |
| **Export time** (100k points) | <5 sec | <2 sec |
| **Memory usage** (100k points) | <500 MB | <300 MB |
| **Max Gaussians** | 1M | 10M |

---

## Dependencies

### Python Packages
```
numpy>=1.26.4
PySide2>=5.15
PyOpenGL>=3.1.5
plyfile>=0.7.4
```

### Maya Requirements
- Maya 2022 or later
- Viewport 2.0 enabled
- Python 3.x support

### Flash3D Requirements
- PyTorch 2.2.2
- CUDA 11.8
- See flash3d/requirements.txt

---

## Troubleshooting

### Common Issues

**1. Plugin fails to load**
- Check Maya Python version matches development
- Verify all dependencies installed
- Check Maya script editor for errors

**2. Point cloud not visible**
- Ensure Viewport 2.0 enabled
- Check LOD not set to 0%
- Verify Gaussian data loaded

**3. Rendered panel blank**
- Check OpenGL context initialized
- Verify shader compilation
- Check console for GL errors

**4. Transform not syncing**
- Verify callback registered
- Check transform monitor running
- Test manual refresh

---

## Next Steps

After completing Alpha:
1. User testing with sample scenes
2. Performance profiling and optimization
3. Documentation and tutorials
4. Beta feature planning
5. Consider C++ viewport upgrade
6. Explore true diff-gaussian-rasterization integration

---

## Resources

- [Flash3D Paper](https://arxiv.org/abs/2406.04343)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Maya Python API 2.0](https://help.autodesk.com/view/MAYAUL/2022/ENU/?guid=Maya_SDK_py_ref_index_html)
- [Viewport 2.0 Override](https://help.autodesk.com/view/MAYAUL/2022/ENU/?guid=Maya_SDK_Viewport_2_0_API_index_html)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-21
**Author:** SplatCraft Development Team

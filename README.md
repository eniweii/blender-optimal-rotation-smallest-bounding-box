# Optimal Rotation (Smallest Bounding Box)

A Blender addon that automatically rotates objects to minimize their axis-aligned bounding box volume. Useful for 3D printing, UV unwrapping, packing, and general scene optimization.

![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange)

## Features

- **Automatic rotation optimization** using PCA with iterative refinement
- **Convex hull preprocessing** for accurate results regardless of internal geometry
- **Batch processing** - optimize multiple selected objects at once
- **Axis constraints** - limit rotation to specific axes (X, Y, Z or any combination)
- **Pivot options** - rotate around geometry center or object origin
- **Align longest axis** - force the longest dimension to align with a specific world axis
- **Bounding box preview** - visualize the current bounding box in the viewport
- **Volume reduction reporting** - see how much the bounding box was reduced

## Installation

### Blender 4.2+ / 5.0 (Extension Format)

1. Download or build the extension zip:
   - **Option A**: Download `optimal_rotation_extension.zip` from releases
   - **Option B**: Clone the repo and run `./build_extension.sh`
2. Open Blender and go to **Edit > Preferences > Get Extensions**
3. Click the dropdown arrow (top right) > **Install from Disk**
4. Select `optimal_rotation_extension.zip`

### Building from Source

```bash
git clone https://github.com/yourusername/blender-optimal-rotation-smallest-bounding-box.git
cd blender-optimal-rotation-smallest-bounding-box
./build_extension.sh
```

This creates `optimal_rotation_extension.zip` ready for installation.

### File Structure

```
optimal_rotation.py      # Main addon code
blender_manifest.toml    # Extension metadata
build_extension.sh       # Build script
```

## Usage

1. Select one or more objects (Mesh, Curve, Surface, Font, or Meta)
2. Open the sidebar in the 3D Viewport (`N` key)
3. Go to the **Item** tab and find the **Optimal Rotation** panel
4. Configure options:
   - **Rotation Axes**: Toggle which axes are allowed to rotate
   - **Pivot Point**: Choose between geometry center or object origin
   - **Align Longest Axis**: Optionally force longest dimension to X, Y, or Z
5. Click **Calculate Optimal Rotation**

### Preview

Use the **Show BBox** button to display the current bounding box wireframe in the viewport. The dimensions and volume will be reported. Click **Clear** to remove the preview.

## Options

| Option | Description |
|--------|-------------|
| **X / Y / Z** | Enable/disable rotation around each axis |
| **Pivot Point** | Geometry Center (default) or Object Origin |
| **Align Longest Axis** | None, X, Y, or Z - forces longest dimension alignment |

## How It Works

1. Extracts world-space vertices from the object (including modifiers)
2. Computes the convex hull to focus on outer geometry only
3. Runs PCA (Principal Component Analysis) to find initial optimal alignment
4. Performs two-pass iterative refinement:
   - Coarse pass: tests rotations in ~11° increments
   - Fine pass: tests rotations in ~1° increments around best result
5. Applies the rotation that produces the smallest bounding box volume

## Supported Object Types

- Mesh
- Curve
- Surface
- Font (Text)
- Metaball

## Requirements

- Blender 4.2 or newer
- NumPy (included with Blender)

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

- **Non-commercial use only** - You may not sell this software or use it in commercial products
- **Attribution required** - You must credit "NERKTEK - Noah Eisenbruch" and link to this repository when using, sharing, or adapting this code

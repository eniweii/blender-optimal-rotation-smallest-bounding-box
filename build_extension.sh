#!/bin/bash
# Build Blender extension zip file

cd "$(dirname "$0")"

ZIP_NAME="optimal_rotation_extension.zip"

# Remove old zip if exists
rm -f "$ZIP_NAME"

# Create temp directory for packaging
TEMP_DIR=$(mktemp -d)
cp optimal_rotation.py "$TEMP_DIR/__init__.py"
cp blender_manifest.toml "$TEMP_DIR/"

# Create zip from temp directory
cd "$TEMP_DIR"
zip -r "$OLDPWD/$ZIP_NAME" __init__.py blender_manifest.toml

# Cleanup
rm -rf "$TEMP_DIR"

echo "Created $ZIP_NAME"

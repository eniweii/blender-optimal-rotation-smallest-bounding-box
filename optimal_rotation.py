bl_info = {
    "name": "Optimal Rotation (Smallest Bounding Box)",
    "author": "Noah Eisenbruch",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Item Tab",
    "description": "Rotates objects to minimize world-space bounding box using convex hull and iterative refinement",
    "category": "Object",
}

import bpy
import bmesh
import mathutils
import math
import numpy as np
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader

# Global for bbox visualization
_bbox_draw_handler = None
_bbox_data = None


class OBJECT_OT_optimal_rotation(bpy.types.Operator):
    """Calculate and apply optimal rotation for smallest bounding box"""
    bl_idname = "object.optimal_rotation"
    bl_label = "Optimal Rotation"
    bl_options = {'REGISTER', 'UNDO'}

    axis_x: bpy.props.BoolProperty(
        name="Rotate X",
        default=True,
        description="Allow rotation around the X axis"
    )
    axis_y: bpy.props.BoolProperty(
        name="Rotate Y",
        default=True,
        description="Allow rotation around the Y axis"
    )
    axis_z: bpy.props.BoolProperty(
        name="Rotate Z",
        default=True,
        description="Allow rotation around the Z axis"
    )

    use_origin_pivot: bpy.props.BoolProperty(
        name="Use Object Origin",
        default=False,
        description="Rotate around object origin instead of geometry center"
    )

    align_longest_to: bpy.props.EnumProperty(
        name="Align Longest To",
        items=[
            ('NONE', "None", "Don't constrain longest axis alignment"),
            ('X', "X Axis", "Align longest dimension to World X"),
            ('Y', "Y Axis", "Align longest dimension to World Y"),
            ('Z', "Z Axis", "Align longest dimension to World Z"),
        ],
        default='NONE',
        description="Align the longest dimension of the bounding box to a specific world axis"
    )

    @classmethod
    def poll(cls, context):
        valid_types = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
        if context.active_object and context.active_object.type in valid_types:
            return True
        # Also allow if any selected object is valid
        for obj in context.selected_objects:
            if obj.type in valid_types:
                return True
        return False

    def execute(self, context):
        active_axes = [self.axis_x, self.axis_y, self.axis_z]
        if not any(active_axes):
            self.report({'WARNING'}, "No axes selected")
            return {'CANCELLED'}

        valid_types = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
        objects_to_process = [obj for obj in context.selected_objects if obj.type in valid_types]

        if not objects_to_process:
            self.report({'ERROR'}, "No valid objects selected")
            return {'CANCELLED'}

        total_reduction = 0
        processed = 0
        depsgraph = context.evaluated_depsgraph_get()

        for obj in objects_to_process:
            result = self.process_object(obj, depsgraph, active_axes)
            if result is not None:
                total_reduction += result
                processed += 1

        if processed > 0:
            avg_reduction = total_reduction / processed
            if processed == 1:
                self.report({'INFO'}, f"Bounding box volume reduced by {avg_reduction:.1f}%")
            else:
                self.report({'INFO'}, f"Processed {processed} objects, avg volume reduction: {avg_reduction:.1f}%")
        else:
            self.report({'WARNING'}, "No objects could be processed")

        return {'FINISHED'}

    def process_object(self, obj, depsgraph, active_axes):
        """Process a single object. Returns reduction percentage or None on failure."""
        obj_eval = obj.evaluated_get(depsgraph)

        try:
            mesh_eval = bpy.data.meshes.new_from_object(obj_eval)
        except Exception:
            return None

        try:
            v_count = len(mesh_eval.vertices)
            if v_count < 4:
                return None

            # Fast vectorized vertex extraction
            coords = np.empty(v_count * 3, dtype=np.float64)
            mesh_eval.vertices.foreach_get("co", coords)
            coords = coords.reshape((-1, 3))

            # Vectorized world transform
            mat = np.array(obj.matrix_world, dtype=np.float64)
            ones = np.ones((v_count, 1), dtype=np.float64)
            coords_4d = np.hstack([coords, ones])
            points = (coords_4d @ mat.T)[:, :3]

            # Convex hull
            bm = bmesh.new()
            for p in points:
                bm.verts.new(p)
            bm.verts.ensure_lookup_table()

            try:
                hull_result = bmesh.ops.convex_hull(bm, input=bm.verts)
                hull_verts = hull_result['geom']
                hull_points_list = [v.co[:] for v in hull_verts if isinstance(v, bmesh.types.BMVert)]
                if not hull_points_list:
                    hull_points = points
                else:
                    hull_points = np.array(hull_points_list)
            except Exception:
                hull_points = points
            finally:
                bm.free()

        finally:
            bpy.data.meshes.remove(mesh_eval)

        if len(hull_points) < 4:
            return None

        # Determine pivot point
        if self.use_origin_pivot:
            pivot = np.array(obj.matrix_world.translation)
        else:
            pivot = np.mean(hull_points, axis=0)

        # Calculate initial bounding box volume
        initial_bbox = self.compute_bbox_volume(hull_points)

        # Route to appropriate alignment method
        if all(active_axes):
            self.apply_optimal_rotation_3d(obj, hull_points, pivot)
        elif sum(active_axes) == 1:
            axis_idx = active_axes.index(True)
            self.apply_optimal_rotation_1d(obj, hull_points, axis_idx, pivot)
        else:
            locked_axis = active_axes.index(False)
            self.apply_optimal_rotation_2d(obj, hull_points, locked_axis, pivot)

        # Calculate final volume
        mat_new = np.array(obj.matrix_world, dtype=np.float64)
        mat_inv = np.linalg.inv(mat)
        local_hull = (np.hstack([hull_points, np.ones((len(hull_points), 1))]) @ mat_inv.T)[:, :3]
        new_world = (np.hstack([local_hull, np.ones((len(local_hull), 1))]) @ mat_new.T)[:, :3]
        final_bbox = self.compute_bbox_volume(new_world)

        reduction = (1 - final_bbox / initial_bbox) * 100 if initial_bbox > 0 else 0
        return reduction

    def compute_bbox_volume(self, points):
        """Compute axis-aligned bounding box volume."""
        min_pt = np.min(points, axis=0)
        max_pt = np.max(points, axis=0)
        dims = max_pt - min_pt
        return dims[0] * dims[1] * dims[2]

    def apply_optimal_rotation_3d(self, obj, points, pivot):
        """Apply PCA-based rotation with two-pass iterative refinement."""
        mean = np.mean(points, axis=0)
        centered_points = points - mean

        # PCA via Covariance
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        sorted_eigenvalues = eigenvalues[idx]

        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        best_rotation = eigenvectors.T
        best_volume = self._compute_rotated_bbox_volume(points, best_rotation)

        # Two-pass refinement for better accuracy
        # Pass 1: Coarse search (-45° to +45°, 9 steps = 11.25° increments)
        for ax in range(3):
            for angle in np.linspace(-np.pi/4, np.pi/4, 9):
                if angle == 0:
                    continue
                axis_vec = np.zeros(3)
                axis_vec[ax] = 1.0
                test_rot = self._axis_angle_to_matrix(axis_vec, angle)
                combined = test_rot @ best_rotation

                vol = self._compute_rotated_bbox_volume(points, combined)
                if vol < best_volume:
                    best_volume = vol
                    best_rotation = combined

        # Pass 2: Fine search around current best (-10° to +10°, 21 steps = 1° increments)
        for ax in range(3):
            for angle in np.linspace(-np.pi/18, np.pi/18, 21):
                if angle == 0:
                    continue
                axis_vec = np.zeros(3)
                axis_vec[ax] = 1.0
                test_rot = self._axis_angle_to_matrix(axis_vec, angle)
                combined = test_rot @ best_rotation

                vol = self._compute_rotated_bbox_volume(points, combined)
                if vol < best_volume:
                    best_volume = vol
                    best_rotation = combined

        # Apply longest axis alignment if requested
        if self.align_longest_to != 'NONE':
            best_rotation = self._align_longest_axis(points, best_rotation, self.align_longest_to)

        rotation_matrix = mathutils.Matrix(best_rotation.tolist()).to_3x3()
        rot_mat_4x4 = rotation_matrix.to_4x4()

        pivot_vec = mathutils.Vector(pivot)
        mw = obj.matrix_world
        mat_trans_inv = mathutils.Matrix.Translation(-pivot_vec)
        mat_trans = mathutils.Matrix.Translation(pivot_vec)

        obj.matrix_world = mat_trans @ rot_mat_4x4 @ mat_trans_inv @ mw

    def _align_longest_axis(self, points, rotation, target_axis_str):
        """Rotate so longest bbox dimension aligns with target world axis."""
        rotated = points @ rotation.T
        min_pt = np.min(rotated, axis=0)
        max_pt = np.max(rotated, axis=0)
        dims = max_pt - min_pt

        # Find which rotated axis has the longest dimension
        longest_idx = np.argmax(dims)
        target_idx = {'X': 0, 'Y': 1, 'Z': 2}[target_axis_str]

        if longest_idx == target_idx:
            return rotation  # Already aligned

        # Swap axes by applying 90° rotation
        # We need to rotate so axis[longest_idx] maps to axis[target_idx]
        swap_rotation = np.eye(3)

        if (longest_idx, target_idx) in [(0, 1), (1, 0)]:
            # Swap X and Y: rotate 90° around Z
            swap_rotation = self._axis_angle_to_matrix(np.array([0, 0, 1]), np.pi/2)
        elif (longest_idx, target_idx) in [(0, 2), (2, 0)]:
            # Swap X and Z: rotate 90° around Y
            swap_rotation = self._axis_angle_to_matrix(np.array([0, 1, 0]), np.pi/2)
        elif (longest_idx, target_idx) in [(1, 2), (2, 1)]:
            # Swap Y and Z: rotate 90° around X
            swap_rotation = self._axis_angle_to_matrix(np.array([1, 0, 0]), np.pi/2)

        return swap_rotation @ rotation

    def _compute_rotated_bbox_volume(self, points, rotation_matrix):
        """Compute bbox volume after applying rotation."""
        rotated = points @ rotation_matrix.T
        min_pt = np.min(rotated, axis=0)
        max_pt = np.max(rotated, axis=0)
        dims = max_pt - min_pt
        return dims[0] * dims[1] * dims[2]

    def _axis_angle_to_matrix(self, axis, angle):
        """Convert axis-angle to 3x3 rotation matrix (Rodrigues)."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def apply_optimal_rotation_1d(self, obj, points, axis_idx, pivot):
        """Optimize rotation around a single axis with two-pass refinement."""
        plane_indices = [i for i in range(3) if i != axis_idx]
        idx_u, idx_v = plane_indices

        points_2d = points[:, [idx_u, idx_v]]

        # 2D PCA
        mean_2d = np.mean(points_2d, axis=0)
        centered = points_2d - mean_2d
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        major_axis = eigvecs[:, 0]

        pca_angle = -math.atan2(major_axis[1], major_axis[0])

        # Two-pass refinement
        # Pass 1: Coarse
        best_angle = pca_angle
        best_area = self._compute_2d_bbox_area(points_2d, pca_angle)

        for offset in np.linspace(-np.pi/4, np.pi/4, 17):
            if offset == 0:
                continue
            test_angle = pca_angle + offset
            area = self._compute_2d_bbox_area(points_2d, test_angle)
            if area < best_area:
                best_area = area
                best_angle = test_angle

        # Pass 2: Fine around best
        refined_angle = best_angle
        for offset in np.linspace(-np.pi/36, np.pi/36, 21):  # -5° to +5°, 0.5° steps
            if offset == 0:
                continue
            test_angle = best_angle + offset
            area = self._compute_2d_bbox_area(points_2d, test_angle)
            if area < best_area:
                best_area = area
                refined_angle = test_angle

        rotation_axis = 'X' if axis_idx == 0 else ('Y' if axis_idx == 1 else 'Z')
        rot_mat = mathutils.Matrix.Rotation(refined_angle, 4, rotation_axis)

        pivot_vec = mathutils.Vector(pivot)
        mw = obj.matrix_world
        mat_trans_inv = mathutils.Matrix.Translation(-pivot_vec)
        mat_trans = mathutils.Matrix.Translation(pivot_vec)

        obj.matrix_world = mat_trans @ rot_mat @ mat_trans_inv @ mw

    def _compute_2d_bbox_area(self, points_2d, angle):
        """Compute 2D bbox area after rotation by angle."""
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = points_2d @ rot.T
        min_pt = np.min(rotated, axis=0)
        max_pt = np.max(rotated, axis=0)
        dims = max_pt - min_pt
        return dims[0] * dims[1]

    def apply_optimal_rotation_2d(self, obj, points, locked_axis, pivot):
        """Optimize rotation in 2 axes with two-pass refinement."""
        active_indices = [i for i in range(3) if i != locked_axis]
        idx_u, idx_v = active_indices

        points_2d = points[:, [idx_u, idx_v]]

        mean_2d = np.mean(points_2d, axis=0)
        centered = points_2d - mean_2d
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        idx = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, idx]
        major_axis = eigvecs[:, 0]

        pca_angle = -math.atan2(major_axis[1], major_axis[0])

        # Two-pass refinement
        best_angle = pca_angle
        best_area = self._compute_2d_bbox_area(points_2d, pca_angle)

        for offset in np.linspace(-np.pi/4, np.pi/4, 17):
            if offset == 0:
                continue
            test_angle = pca_angle + offset
            area = self._compute_2d_bbox_area(points_2d, test_angle)
            if area < best_area:
                best_area = area
                best_angle = test_angle

        refined_angle = best_angle
        for offset in np.linspace(-np.pi/36, np.pi/36, 21):
            if offset == 0:
                continue
            test_angle = best_angle + offset
            area = self._compute_2d_bbox_area(points_2d, test_angle)
            if area < best_area:
                best_area = area
                refined_angle = test_angle

        rotation_axis = 'X' if locked_axis == 0 else ('Y' if locked_axis == 1 else 'Z')
        rot_mat = mathutils.Matrix.Rotation(refined_angle, 4, rotation_axis)

        pivot_vec = mathutils.Vector(pivot)
        mw = obj.matrix_world
        mat_trans_inv = mathutils.Matrix.Translation(-pivot_vec)
        mat_trans = mathutils.Matrix.Translation(pivot_vec)

        obj.matrix_world = mat_trans @ rot_mat @ mat_trans_inv @ mw


class OBJECT_OT_preview_bbox(bpy.types.Operator):
    """Preview the bounding box without applying rotation"""
    bl_idname = "object.preview_optimal_bbox"
    bl_label = "Preview Bounding Box"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        valid_types = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
        return context.active_object and context.active_object.type in valid_types

    def execute(self, context):
        global _bbox_draw_handler, _bbox_data

        obj = context.active_object
        if not obj:
            return {'CANCELLED'}

        # Get bbox corners
        bbox_corners = self.get_world_bbox_corners(obj, context)
        if bbox_corners is None:
            self.report({'ERROR'}, "Could not compute bounding box")
            return {'CANCELLED'}

        # Store bbox data for drawing
        _bbox_data = {
            'corners': bbox_corners,
            'color': (0.0, 1.0, 0.5, 0.8),  # Green
        }

        # Add draw handler if not already added
        if _bbox_draw_handler is None:
            _bbox_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                draw_bbox_callback, (), 'WINDOW', 'POST_VIEW'
            )

        # Force redraw
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        # Compute and report dimensions
        min_pt = np.min(bbox_corners, axis=0)
        max_pt = np.max(bbox_corners, axis=0)
        dims = max_pt - min_pt
        volume = dims[0] * dims[1] * dims[2]

        self.report({'INFO'}, f"BBox: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}, Volume: {volume:.4f}")

        return {'FINISHED'}

    def get_world_bbox_corners(self, obj, context):
        """Get world-space bounding box corners."""
        depsgraph = context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)

        try:
            mesh_eval = bpy.data.meshes.new_from_object(obj_eval)
        except Exception:
            return None

        try:
            v_count = len(mesh_eval.vertices)
            if v_count < 1:
                return None

            coords = np.empty(v_count * 3, dtype=np.float64)
            mesh_eval.vertices.foreach_get("co", coords)
            coords = coords.reshape((-1, 3))

            mat = np.array(obj.matrix_world, dtype=np.float64)
            ones = np.ones((v_count, 1), dtype=np.float64)
            coords_4d = np.hstack([coords, ones])
            points = (coords_4d @ mat.T)[:, :3]

            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)

            # Generate 8 corners of AABB
            corners = np.array([
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ])
            return corners

        finally:
            bpy.data.meshes.remove(mesh_eval)


class OBJECT_OT_clear_bbox_preview(bpy.types.Operator):
    """Clear the bounding box preview"""
    bl_idname = "object.clear_bbox_preview"
    bl_label = "Clear Preview"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global _bbox_draw_handler, _bbox_data

        if _bbox_draw_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_bbox_draw_handler, 'WINDOW')
            _bbox_draw_handler = None

        _bbox_data = None

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        return {'FINISHED'}


def draw_bbox_callback():
    """Draw bounding box wireframe in viewport."""
    global _bbox_data

    if _bbox_data is None:
        return

    corners = _bbox_data['corners']
    color = _bbox_data['color']

    # Define edges (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    vertices = []
    for e in edges:
        vertices.append(tuple(corners[e[0]]))
        vertices.append(tuple(corners[e[1]]))

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": vertices})

    gpu.state.line_width_set(2.0)
    gpu.state.blend_set('ALPHA')

    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


class OptimalRotationSettings(bpy.types.PropertyGroup):
    axis_x: bpy.props.BoolProperty(
        name="Limit X",
        default=True,
        description="Allow rotation around World X axis"
    )
    axis_y: bpy.props.BoolProperty(
        name="Limit Y",
        default=True,
        description="Allow rotation around World Y axis"
    )
    axis_z: bpy.props.BoolProperty(
        name="Limit Z",
        default=True,
        description="Allow rotation around World Z axis"
    )
    use_origin_pivot: bpy.props.BoolProperty(
        name="Use Object Origin",
        default=False,
        description="Rotate around object origin instead of geometry center"
    )
    align_longest_to: bpy.props.EnumProperty(
        name="Align Longest To",
        items=[
            ('NONE', "None", "Don't constrain longest axis alignment"),
            ('X', "X Axis", "Align longest dimension to World X"),
            ('Y', "Y Axis", "Align longest dimension to World Y"),
            ('Z', "Z Axis", "Align longest dimension to World Z"),
        ],
        default='NONE',
        description="Align the longest dimension of the bounding box to a specific world axis"
    )


class VIEW3D_PT_optimal_rotation(bpy.types.Panel):
    """Panel for Optimal Rotation"""
    bl_label = "Optimal Rotation"
    bl_idname = "VIEW3D_PT_optimal_rotation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Item'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.optimal_rotation_settings

        # Rotation Axes
        col = layout.column(align=True)
        col.label(text="Rotation Axes:")
        row = col.row(align=True)
        row.prop(settings, "axis_x", toggle=True, text="X")
        row.prop(settings, "axis_y", toggle=True, text="Y")
        row.prop(settings, "axis_z", toggle=True, text="Z")

        layout.separator()

        # Pivot Point
        col = layout.column(align=True)
        col.label(text="Pivot Point:")
        col.prop(settings, "use_origin_pivot", text="Object Origin" if settings.use_origin_pivot else "Geometry Center", toggle=True)

        layout.separator()

        # Align Longest Axis
        col = layout.column(align=True)
        col.label(text="Align Longest Axis:")
        col.prop(settings, "align_longest_to", text="")

        layout.separator()

        # Main operator button
        op = layout.operator("object.optimal_rotation", text="Calculate Optimal Rotation")
        op.axis_x = settings.axis_x
        op.axis_y = settings.axis_y
        op.axis_z = settings.axis_z
        op.use_origin_pivot = settings.use_origin_pivot
        op.align_longest_to = settings.align_longest_to

        # Selection info
        valid_types = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
        valid_count = sum(1 for obj in context.selected_objects if obj.type in valid_types)
        if valid_count > 1:
            layout.label(text=f"{valid_count} objects selected", icon='INFO')

        layout.separator()

        # Preview section
        box = layout.box()
        box.label(text="Preview:", icon='HIDE_OFF')
        row = box.row(align=True)
        row.operator("object.preview_optimal_bbox", text="Show BBox", icon='CUBE')
        row.operator("object.clear_bbox_preview", text="Clear", icon='X')


def register():
    bpy.utils.register_class(OBJECT_OT_optimal_rotation)
    bpy.utils.register_class(OBJECT_OT_preview_bbox)
    bpy.utils.register_class(OBJECT_OT_clear_bbox_preview)
    bpy.utils.register_class(OptimalRotationSettings)
    bpy.utils.register_class(VIEW3D_PT_optimal_rotation)
    bpy.types.Scene.optimal_rotation_settings = bpy.props.PointerProperty(type=OptimalRotationSettings)


def unregister():
    global _bbox_draw_handler, _bbox_data

    # Clean up draw handler
    if _bbox_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_bbox_draw_handler, 'WINDOW')
        _bbox_draw_handler = None
    _bbox_data = None

    del bpy.types.Scene.optimal_rotation_settings
    bpy.utils.unregister_class(VIEW3D_PT_optimal_rotation)
    bpy.utils.unregister_class(OptimalRotationSettings)
    bpy.utils.unregister_class(OBJECT_OT_clear_bbox_preview)
    bpy.utils.unregister_class(OBJECT_OT_preview_bbox)
    bpy.utils.unregister_class(OBJECT_OT_optimal_rotation)


if __name__ == "__main__":
    register()

bl_info = {
    "name": "Optimal Rotation (Smallest Bounding Box)",
    "author": "Noah Eisenbruch",
    "version": (1, 5),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Item Tab",
    "description": "Aligns mesh vertex data to minimise the local bounding box (full 3D PCA)",
    "category": "Object",
}

import bpy
import bmesh
import mathutils
import math
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _convex_hull_pts(points):
    """
    Convex hull of points (N x 3 ndarray).
    Returns (M x 3) hull vertices, or points on failure.
    """
    bm = bmesh.new()
    try:
        for p in points:
            bm.verts.new(p)
        bm.verts.ensure_lookup_table()
        result = bmesh.ops.convex_hull(bm, input=bm.verts)
        hull = np.array([v.co[:] for v in result['geom']
                         if isinstance(v, bmesh.types.BMVert)])
        return hull if len(hull) >= 4 else points
    except Exception:
        return points
    finally:
        bm.free()


def _bbox_volume(pts):
    d = pts.max(axis=0) - pts.min(axis=0)
    return float(d[0] * d[1] * d[2])


def _aa_mat(axis, angle):
    """Rodrigues axis-angle to 3x3 rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[     0,  -axis[2],  axis[1]],
                  [ axis[2],      0,  -axis[0]],
                  [-axis[1],  axis[0],      0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _find_best_rotation_3d(hull_pts):
    """
    Find the 3x3 rotation matrix R that minimises the AABB volume of
    (hull_pts @ R.T).  R rows are the new local basis vectors.

    Why full 3D and not Z-only:
        Z-only rotation in local space only removes tilt in the local XY plane.
        When an object has non-zero X or Y world rotation its stored vertex
        coordinates can be misaligned along all three axes, making the local
        AABB much larger than necessary. Full 3D PCA finds the unique rotation
        that aligns the geometry's principal axes with the coordinate axes,
        giving the tightest possible axis-aligned box in local space.
        The result is ALWAYS flat relative to the object (the six faces of the
        local AABB are always parallel to local X/Y/Z) -- this property comes
        from working on the datablock directly, not from restricting axes.

    Algorithm:
        1. Full 3D PCA                         -- analytically optimal seed.
        2. Coarse axis-angle refinement  +-45 deg / 9 steps per axis.
        3. Fine   axis-angle refinement  +-10 deg / 21 steps per axis.
        4. Deterministic sign convention       -- matches replacer pipeline.
    """
    mean     = hull_pts.mean(axis=0)
    centered = hull_pts - mean
    cov      = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ascending eigenvalue sort:
    #   smallest -> col 0 -> row 0 after .T -> local X (thinnest dimension)
    #   largest  -> col 2 -> row 2 after .T -> local Z (longest dimension)
    # This matches the axis convention expected by the replacer pipeline.
    idx         = eigenvalues.argsort()
    eigenvectors = eigenvectors[:, idx]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    best_rot = eigenvectors.T       # rows are new basis vectors
    best_vol = _bbox_volume(hull_pts @ best_rot.T)

    # Pass 1 -- coarse: +-45 deg around each local axis
    for ax in range(3):
        axis_vec = np.zeros(3); axis_vec[ax] = 1.0
        for angle in np.linspace(-math.pi / 4, math.pi / 4, 9):
            if angle == 0:
                continue
            test = _aa_mat(axis_vec, angle) @ best_rot
            vol  = _bbox_volume(hull_pts @ test.T)
            if vol < best_vol:
                best_vol, best_rot = vol, test

    # Pass 2 -- fine: +-10 deg around each local axis
    for ax in range(3):
        axis_vec = np.zeros(3); axis_vec[ax] = 1.0
        for angle in np.linspace(-math.pi / 18, math.pi / 18, 21):
            if angle == 0:
                continue
            test = _aa_mat(axis_vec, angle) @ best_rot
            vol  = _bbox_volume(hull_pts @ test.T)
            if vol < best_vol:
                best_vol, best_rot = vol, test

    # Deterministic sign convention so the replacer's identity axis mapping holds:
    #   Row 0 (smallest eigenvalue / thinnest): largest-magnitude component positive.
    #   Row 2 (largest  eigenvalue / longest):  largest-magnitude component positive.
    #   Row 1 (medium): cross(row2, row0) -- guarantees right-handedness.
    for row_i in (0, 2):
        primary = int(np.argmax(np.abs(best_rot[row_i])))
        if best_rot[row_i, primary] < 0:
            best_rot[row_i] *= -1
    row1  = np.cross(best_rot[2], best_rot[0])
    norm1 = np.linalg.norm(row1)
    if norm1 > 1e-8:
        best_rot[1] = row1 / norm1
    elif np.linalg.det(best_rot) < 0:
        best_rot[1] *= -1

    return best_rot


def _snap_longest_xy(hull_pts, rot, target_str):
    """
    After optimal rotation, optionally snap the longest XY dimension to
    target_str ('X' or 'Y') by adding a 90 deg Z rotation.
    The Z (longest overall) axis is left untouched.
    """
    dims       = (hull_pts @ rot.T).ptp(axis=0)   # [dx, dy, dz]
    longest_xy = 0 if dims[0] >= dims[1] else 1
    target_idx = 0 if target_str == 'X' else 1
    if longest_xy == target_idx:
        return rot
    z90 = _aa_mat(np.array([0., 0., 1.]), math.pi / 2)
    return z90 @ rot


def _process_datablock(mesh, basis_snap, align_longest_to='NONE'):
    """
    Core routine shared by both operators.

    Rotates mesh vertex data so the geometry's principal axes align with local
    X/Y/Z, giving the tightest possible local AABB.  All objects referencing
    this datablock are compensated via matrix_basis so world-space position and
    orientation are completely unchanged.

    Works entirely in local (datablock) space -- the object's world transform
    is never read or written.  Correct regardless of view layer membership,
    parent hierarchy, or how the object is oriented in the world.

    Returns the % AABB volume reduction, or None on failure / skip.

    Proof of neutrality for every referencing object O:
        world_pos  = parent @ matrix_basis_old @ old_local_co
        new_co     = R @ old_local_co
        new_basis  = matrix_basis_old @ R^-1        (R^-1 = R.T, R is orthonormal)
        world_pos' = parent @ new_basis @ new_co
                   = parent @ (matrix_basis_old @ R.T) @ (R @ old_local_co)
                   = parent @ matrix_basis_old @ old_local_co  =  world_pos  checkmark
    """
    v = len(mesh.vertices)
    if v < 4:
        return None

    raw = np.empty(v * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", raw)
    points = raw.reshape((-1, 3))

    initial_vol = _bbox_volume(points)
    if initial_vol <= 0:
        return None

    hull_pts = _convex_hull_pts(points)
    if len(hull_pts) < 4:
        return None

    rot = _find_best_rotation_3d(hull_pts)

    if align_longest_to != 'NONE':
        rot = _snap_longest_xy(hull_pts, rot, align_longest_to)

    new_coords = (points @ rot.T).astype(np.float32)

    final_vol = _bbox_volume(new_coords)

    # Skip if this rotation produces no meaningful improvement
    # (geometry was already well-aligned, numerical noise only)
    if initial_vol > 0 and (initial_vol - final_vol) / initial_vol < 1e-4:
        return 0.0

    # Compensate matrix_basis for every object using this datablock.
    # matrix_basis is the raw stored transform -- always valid, no depsgraph needed.
    rot_inv_4x4 = mathutils.Matrix(rot.tolist()).to_3x3().transposed().to_4x4()
    for obj in bpy.data.objects:
        if obj.data is not mesh:
            continue
        snap = basis_snap.get(obj.name, obj.matrix_basis.copy())
        obj.matrix_basis = snap @ rot_inv_4x4

    mesh.vertices.foreach_set("co", new_coords.ravel())
    mesh.update()

    return (1.0 - final_vol / initial_vol) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Object Operator
#  Processes only the datablocks of the selected objects.
# ══════════════════════════════════════════════════════════════════════════════

class OBJECT_OT_optimal_rotation(bpy.types.Operator):
    """Minimise the local bounding box of selected objects.
Rotates vertex data so geometry principal axes align with local X/Y/Z.
Objects stay in place; only their mesh data changes (full 3D alignment)."""
    bl_idname  = "object.optimal_rotation"
    bl_label   = "Optimal Rotation"
    bl_options = {'REGISTER', 'UNDO'}

    align_longest_to: bpy.props.EnumProperty(
        name="Align Longest To",
        items=[
            ('NONE', "None",   "Don't constrain longest-axis alignment"),
            ('X',    "X Axis", "Snap longest XY dimension to local X"),
            ('Y',    "Y Axis", "Snap longest XY dimension to local Y"),
        ],
        default='NONE',
    )

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        # Collect unique datablocks from selected mesh objects.
        selected_meshes = {obj.data for obj in context.selected_objects
                           if obj.type == 'MESH' and obj.data is not None}
        if not selected_meshes:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        # Snapshot matrix_basis for ALL objects sharing these datablocks
        # (not just selected ones -- shared users must be compensated too).
        basis_snap = {
            obj.name: obj.matrix_basis.copy()
            for obj in bpy.data.objects
            if obj.type == 'MESH' and obj.data in selected_meshes
        }

        total_red, processed = 0.0, 0
        for mesh in selected_meshes:
            r = _process_datablock(mesh, basis_snap, self.align_longest_to)
            if r is not None:
                total_red += r
                processed += 1

        if processed == 0:
            self.report({'WARNING'}, "No meshes could be processed")
            return {'CANCELLED'}

        avg = total_red / processed
        msg = (f"Bounding box volume reduced by {avg:.1f}%"
               if processed == 1
               else f"Processed {processed} meshes, avg reduction: {avg:.1f}%")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Datablock Operator
#  Same core logic, applied to every mesh datablock in the file.
# ══════════════════════════════════════════════════════════════════════════════

class OBJECT_OT_optimal_rotation_datablocks(bpy.types.Operator):
    """Apply optimal 3D rotation to every mesh datablock in the file.
Identical algorithm to per-object, just applied to all meshes at once."""
    bl_idname  = "object.optimal_rotation_datablocks"
    bl_label   = "Process All Mesh Datablocks"
    bl_options = {'REGISTER', 'UNDO'}

    align_longest_to: bpy.props.EnumProperty(
        name="Align Longest To",
        items=[
            ('NONE', "None",   "Don't constrain longest-axis alignment"),
            ('X',    "X Axis", "Snap longest XY dimension to local X"),
            ('Y',    "Y Axis", "Snap longest XY dimension to local Y"),
        ],
        default='NONE',
    )

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.meshes)

    def execute(self, context):
        # Single global snapshot before touching any vertex data.
        basis_snap = {
            obj.name: obj.matrix_basis.copy()
            for obj in bpy.data.objects
            if obj.type == 'MESH'
        }

        all_meshes         = list(bpy.data.meshes)
        total              = len(all_meshes)
        processed, skipped = 0, 0
        total_red          = 0.0

        print(f"\nOptimal Rotation -- processing {total} mesh datablocks ...")

        for i, mesh in enumerate(all_meshes, 1):
            tag = f"[{i}/{total}] {mesh.name!r:40s}"
            r = _process_datablock(mesh, basis_snap, self.align_longest_to)
            if r is None:
                print(f"{tag} -- skipped")
                skipped += 1
            else:
                print(f"{tag} -- reduced by {r:.1f}%")
                total_red += r
                processed += 1

        avg = total_red / processed if processed else 0.0
        msg = (f"Processed {processed} datablocks (skipped {skipped}). "
               f"Avg bbox volume reduction: {avg:.1f}%")
        print(f"\n{msg}\n")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ══════════════════════════════════════════════════════════════════════════════
#  Viewport Bounds Toggle Operators
# ══════════════════════════════════════════════════════════════════════════════

class OBJECT_OT_toggle_bounds_active(bpy.types.Operator):
    """Toggle Viewport Display > Bounds for the active object"""
    bl_idname  = "object.toggle_bounds_active"
    bl_label   = "Toggle Bounds (Active)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        obj = context.active_object
        obj.show_bounds         = not obj.show_bounds
        obj.display_bounds_type = 'BOX'
        self.report({'INFO'},
                    f"Bounds {'on' if obj.show_bounds else 'off'} for '{obj.name}'")
        return {'FINISHED'}


class OBJECT_OT_toggle_bounds_all(bpy.types.Operator):
    """Toggle Viewport Display > Bounds on ALL objects.
Turns all off if any are on; otherwise turns all on."""
    bl_idname  = "object.toggle_bounds_all"
    bl_label   = "Toggle Bounds (All)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(bpy.data.objects)

    def execute(self, context):
        objects = list(bpy.data.objects)
        target  = not any(o.show_bounds for o in objects)
        for obj in objects:
            obj.show_bounds         = target
            obj.display_bounds_type = 'BOX'
        self.report({'INFO'},
                    f"Bounds {'on' if target else 'off'} for {len(objects)} objects")
        return {'FINISHED'}


# ══════════════════════════════════════════════════════════════════════════════
#  Settings & Panel
# ══════════════════════════════════════════════════════════════════════════════

class OptimalRotationSettings(bpy.types.PropertyGroup):
    align_longest_to: bpy.props.EnumProperty(
        name="Align Longest To",
        items=[
            ('NONE', "None",   "Don't constrain longest-axis alignment"),
            ('X',    "X Axis", "Snap longest XY dimension to local X"),
            ('Y',    "Y Axis", "Snap longest XY dimension to local Y"),
        ],
        default='NONE',
    )


class VIEW3D_PT_optimal_rotation(bpy.types.Panel):
    bl_label       = "Optimal Rotation"
    bl_idname      = "VIEW3D_PT_optimal_rotation"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Item'

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.optimal_rotation_settings

        col = layout.column(align=True)
        col.label(text="Align Longest XY To:")
        col.prop(settings, "align_longest_to", text="")

        layout.separator()

        # ---- Per-object (selected datablocks) --------------------------------
        box = layout.box()
        box.label(text="Selected Objects:", icon='OBJECT_DATA')
        box.label(text="Full 3D PCA -- rotates vertex data.", icon='INFO')

        op = box.operator("object.optimal_rotation", text="Align Selected")
        op.align_longest_to = settings.align_longest_to

        n = sum(1 for o in context.selected_objects if o.type == 'MESH')
        if n > 0:
            box.label(text=f"{n} mesh object(s) selected", icon='BLANK1')

        layout.separator()

        # ---- Batch (all datablocks) ------------------------------------------
        box_db = layout.box()
        box_db.label(text="All Mesh Datablocks:", icon='MESH_DATA')
        box_db.label(text="Same algorithm, every mesh in file.", icon='INFO')

        op_db = box_db.operator(
            "object.optimal_rotation_datablocks",
            text="Process All Datablocks",
            icon='WORLD',
        )
        op_db.align_longest_to = settings.align_longest_to
        box_db.label(text=f"{len(bpy.data.meshes)} mesh datablocks in file",
                     icon='BLANK1')

        layout.separator()

        # ---- Viewport bounds -------------------------------------------------
        box_b = layout.box()
        box_b.label(text="Viewport Bounds:", icon='CUBE')
        row = box_b.row(align=True)
        row.operator("object.toggle_bounds_active",
                     text="Active Object", icon='OBJECT_DATA')
        row.operator("object.toggle_bounds_all",
                     text="All Objects",   icon='WORLD')


# ══════════════════════════════════════════════════════════════════════════════
#  Registration
# ══════════════════════════════════════════════════════════════════════════════

_CLASSES = (
    OBJECT_OT_optimal_rotation,
    OBJECT_OT_optimal_rotation_datablocks,
    OBJECT_OT_toggle_bounds_active,
    OBJECT_OT_toggle_bounds_all,
    OptimalRotationSettings,
    VIEW3D_PT_optimal_rotation,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.optimal_rotation_settings = bpy.props.PointerProperty(
        type=OptimalRotationSettings
    )


def unregister():
    if hasattr(bpy.types.Scene, 'optimal_rotation_settings'):
        del bpy.types.Scene.optimal_rotation_settings
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()

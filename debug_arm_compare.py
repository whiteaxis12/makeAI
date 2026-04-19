"""
debug_arm_compare.py
BVHとanimation_output.fbxの腕の回転値を比較する
"""
import bpy
import os

ANIMATION_FBX_PATH = "data/outputs/animation_output.fbx"
BVH_PATH           = "data/outputs/sample_normalized.bvh"

TARGET_BONES = {
    "LeftShoulder":  "mixamorig:LeftShoulder",
    "LeftArm":       "mixamorig:LeftArm",
    "LeftForeArm":   "mixamorig:LeftForeArm",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightArm":      "mixamorig:RightArm",
    "RightForeArm":  "mixamorig:RightForeArm",
}

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_fbx(path):
    abs_path = os.path.abspath(path)
    bpy.ops.import_scene.fbx(
        filepath=abs_path,
        use_anim=True,          # ← アニメーション有効
        anim_offset=0,          # ← オフセットなし
        ignore_leaf_bones=False,
        force_connect_children=False,
    )
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None

def import_bvh(path):
    abs_path = os.path.abspath(path)
    bpy.ops.import_anim.bvh(
        filepath=abs_path,
        axis_forward='Z',
        axis_up='Y',
        rotate_mode='NATIVE',
        global_scale=0.01
    )
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            return obj
    return None

def compare(fbx_armature, bvh_armature, frames=[1, 30, 60]):
    if fbx_armature.animation_data:
        action = fbx_armature.animation_data.action
        if action:
            print(f"[INFO] FBXアクション: {action.name} フレーム範囲: {action.frame_range}")
            print(f"[INFO] FCurve数: {len(action.fcurves)}")

    for frame in frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        print(f"\n{'='*80}")
        print(f"フレーム {frame}")
        print(f"{'='*80}")
        print(f"{'ボーン':20s} {'BVH euler(ZXY)':35s} {'FBX euler(変換後)':35s} {'差分':30s}")
        print(f"{'-'*120}")

        for bvh_name, fbx_name in TARGET_BONES.items():
            bvh_bone = bvh_armature.pose.bones.get(bvh_name)
            fbx_bone = fbx_armature.pose.bones.get(fbx_name)

            if bvh_bone is None or fbx_bone is None:
                continue

            # BVHのオイラー角（ZXY）
            bvh_e   = bvh_bone.rotation_euler
            bvh_str = f"({bvh_e.x*57.3:7.2f},{bvh_e.y*57.3:7.2f},{bvh_e.z*57.3:7.2f})"

            # FBXはクォータニオンのままで取得してオイラーに変換
            fbx_bone.rotation_mode = 'QUATERNION'
            fbx_quat = fbx_bone.rotation_quaternion.copy()
            fbx_euler = fbx_quat.to_euler('ZXY')
            fbx_str  = f"({fbx_euler.x*57.3:7.2f},{fbx_euler.y*57.3:7.2f},{fbx_euler.z*57.3:7.2f})"

            # 差分
            dx = (fbx_euler.x - bvh_e.x) * 57.3
            dy = (fbx_euler.y - bvh_e.y) * 57.3
            dz = (fbx_euler.z - bvh_e.z) * 57.3
            diff_str = f"({dx:6.2f},{dy:6.2f},{dz:6.2f})"

            print(f"{bvh_name:20s} BVH={bvh_str} FBX={fbx_str} diff={diff_str}")

def main():
    print("="*80)
    print("BVH vs animation_output.fbx 腕の回転値比較")
    print("="*80)

    clear_scene()

    # animation_output.fbxをインポート
    fbx_armature = import_fbx(ANIMATION_FBX_PATH)
    if fbx_armature is None:
        print("[ERROR] FBXアーマチュアが見つかりません")
        return
    print(f"[OK] FBX: {fbx_armature.name}")

    # BVHをインポート
    bvh_armature = import_bvh(BVH_PATH)
    if bvh_armature is None:
        print("[ERROR] BVHアーマチュアが見つかりません")
        return
    print(f"[OK] BVH: {bvh_armature.name}")

    compare(fbx_armature, bvh_armature, frames=[1, 30, 60])

main()
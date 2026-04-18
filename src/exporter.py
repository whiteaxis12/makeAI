import numpy as np
from datetime import datetime

class BVHExporter:
    """MixamoボーンデータをBVH形式で出力"""

    # BVHのボーン階層定義（Mixamo標準）
    HIERARCHY = [
        ("Hips",        None,           [0, 0, 0]),
        ("Spine",       "Hips",         [0, 10, 0]),
        ("Spine1",      "Spine",        [0, 10, 0]),
        ("Spine2",      "Spine1",       [0, 10, 0]),
        ("Neck",        "Spine2",       [0, 10, 0]),
        ("Head",        "Neck",         [0, 10, 0]),

        ("LeftArm",     "Spine2",       [10, 0, 0]),
        ("LeftForeArm", "LeftArm",      [20, 0, 0]),
        ("LeftHand",    "LeftForeArm",  [20, 0, 0]),

        ("RightArm",    "Spine2",       [-10, 0, 0]),
        ("RightForeArm","RightArm",     [-20, 0, 0]),
        ("RightHand",   "RightForeArm", [-20, 0, 0]),

        ("LeftUpLeg",   "Hips",         [5, -10, 0]),
        ("LeftLeg",     "LeftUpLeg",    [0, -20, 0]),
        ("LeftFoot",    "LeftLeg",      [0, -20, 0]),
        ("LeftToeBase", "LeftFoot",     [0, -5,  5]),

        ("RightUpLeg",  "Hips",         [-5, -10, 0]),
        ("RightLeg",    "RightUpLeg",   [0, -20, 0]),
        ("RightFoot",   "RightLeg",     [0, -20, 0]),
        ("RightToeBase","RightFoot",    [0, -5,  5]),
    ]

    # 信頼度がこの値以下のボーンはデフォルト回転を使用
    CONFIDENCE_THRESHOLD = 0.3

    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate

    def _build_hierarchy_text(self) -> str:
        """BVHのHIERARCHYセクションを生成"""
        lines = ["HIERARCHY"]
        indent = ""
        stack = []

        for bone_name, parent, offset in self.HIERARCHY:
            # ルートボーン
            if parent is None:
                lines.append(f"ROOT {bone_name}")
                lines.append("{")
                indent = "\t"
            else:
                # 親が変わったらEnd Siteと閉じ括弧を処理
                while stack and stack[-1] != parent:
                    closed = stack.pop()
                    indent = "\t" * len(stack)
                    # End Site追加
                    lines.append(f"{indent}\tEnd Site")
                    lines.append(f"{indent}\t{{")
                    lines.append(f"{indent}\t\tOFFSET 0.00 0.00 0.00")
                    lines.append(f"{indent}\t}}")
                    lines.append(f"{indent}}}")

                indent = "\t" * (len(stack) + 1)
                lines.append(f"{indent}JOINT {bone_name}")
                lines.append(f"{indent}{{")
                indent += "\t"

            lines.append(f"{indent}OFFSET {offset[0]:.2f} {offset[1]:.2f} {offset[2]:.2f}")

            if parent is None:
                lines.append(f"{indent}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
            else:
                lines.append(f"{indent}CHANNELS 3 Zrotation Xrotation Yrotation")

            stack.append(bone_name)

        # 残りのEnd Siteと閉じ括弧
        while stack:
            indent = "\t" * (len(stack) - 1)
            lines.append(f"{indent}\tEnd Site")
            lines.append(f"{indent}\t{{")
            lines.append(f"{indent}\t\tOFFSET 0.00 0.00 0.00")
            lines.append(f"{indent}\t}}")
            lines.append(f"{indent}}}")
            stack.pop()

        return "\n".join(lines)

    def _build_motion_text(self, frames: list[dict]) -> str:
        """BVHのMOTIONセクションを生成"""
        lines = ["MOTION"]
        lines.append(f"Frames: {len(frames)}")
        lines.append(f"Frame Time: {1.0 / self.frame_rate:.6f}")

        for frame_data in frames:
            values = []

            for i, (bone_name, parent, _) in enumerate(self.HIERARCHY):
                bone = frame_data.get(bone_name, {})
                rotation = bone.get("rotation", [0, 0, 0])
                confidence = bone.get("confidence", 0.0)

                # 信頼度が低い場合はデフォルト値（0度）を使用
                if confidence < self.CONFIDENCE_THRESHOLD:
                    rotation = [0, 0, 0]

                if parent is None:
                    # ルートボーンは位置+回転
                    position = bone.get("position", [0, 0, 0])
                    values.extend([
                        position[0] * 100,  # メートル→センチ変換
                        position[1] * 100,
                        position[2] * 100,
                        rotation[2],  # Zrotation
                        rotation[0],  # Xrotation
                        rotation[1],  # Yrotation
                    ])
                else:
                    values.extend([
                        rotation[2],  # Zrotation
                        rotation[0],  # Xrotation
                        rotation[1],  # Yrotation
                    ])

            lines.append(" ".join(f"{v:.4f}" for v in values))

        return "\n".join(lines)

    def export(self, bone_data: dict, output_path: str, frames: list[dict] = None):
        """
        BVHファイルを出力
        bone_data: converter.pyの出力
        output_path: 出力先パス
        frames: 複数フレームの場合のリスト（Noneなら1フレーム）
        """
        if frames is None:
            frames = [bone_data]

        hierarchy_text = self._build_hierarchy_text()
        motion_text    = self._build_motion_text(frames)

        bvh_content = hierarchy_text + "\n" + motion_text

        with open(output_path, "w") as f:
            f.write(bvh_content)

        print(f"[OK] BVHファイルを出力しました: {output_path}")
        print(f"     フレーム数: {len(frames)}")
        print(f"     ボーン数:   {len(self.HIERARCHY)}")
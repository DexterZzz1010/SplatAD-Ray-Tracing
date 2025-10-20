#!/usr/bin/env python3
"""
纯NumPy版本的Dummy数据生成器
Linus风格: 零依赖,能跑就行!
"""

import numpy as np
from pathlib import Path
import json


class MinimalDummyNuScenes:
    """
    最小化的dummy数据生成器
    哲学: 只依赖numpy,谁都能跑
    """
    
    def __init__(self, num_frames=10, num_cameras=6):
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.img_h, self.img_w = 900, 1600
        
    def generate_cameras(self):
        """生成相机数据"""
        print(f"\n[生成] {self.num_frames}帧 x {self.num_cameras}相机 = {self.num_frames * self.num_cameras}个相机pose")
        
        data = {
            'poses': [],
            'intrinsics': [],
            'timestamps': [],
            'image_filenames': []
        }
        
        for frame in range(self.num_frames):
            t = frame / self.num_frames * 2 * np.pi
            
            for cam in range(self.num_cameras):
                # 圆形轨迹
                angle = t + cam * (2 * np.pi / self.num_cameras)
                x = 5.0 * np.cos(angle)
                y = 5.0 * np.sin(angle)
                z = 1.5
                
                # Pose: 简单的lookat矩阵
                forward = np.array([-x, -y, 0])
                forward = forward / (np.linalg.norm(forward) + 1e-10)
                right = np.cross(forward, np.array([0, 0, 1]))
                right = right / (np.linalg.norm(right) + 1e-10)
                up = np.cross(right, forward)
                
                pose = np.eye(4)
                pose[:3, 0] = right
                pose[:3, 1] = up
                pose[:3, 2] = forward
                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = z
                
                data['poses'].append(pose)
                
                # 内参
                fx, fy = 1266.4, 1266.4
                cx, cy = self.img_w / 2, self.img_h / 2
                data['intrinsics'].append([fx, fy, cx, cy])
                data['timestamps'].append(frame * 0.5)
                data['image_filenames'].append(f"cam_{cam}/frame_{frame:04d}.jpg")
        
        return data
    
    def generate_lidars(self):
        """生成激光雷达点云"""
        print(f"[生成] {self.num_frames}帧点云数据")
        
        data = {
            'frames': [],
            'timestamps': []
        }
        
        for frame in range(self.num_frames):
            # 地面 + 物体
            num_points = 10000
            
            # 地面
            ground = np.random.randn(num_points // 2, 3)
            ground[:, 2] = -1.5 + np.random.randn(num_points // 2) * 0.1
            ground[:, :2] *= 20
            
            # 物体(3个车)
            objects = []
            for i in range(3):
                center = np.array([5 + i * 3, 0, 0.5])
                obj = np.random.randn(num_points // 6, 3) * 0.5 + center
                objects.append(obj)
            
            points = np.vstack([ground] + objects)
            intensity = np.random.rand(len(points), 1) * 255
            
            # 合并 [x, y, z, intensity]
            pointcloud = np.hstack([points, intensity])
            
            data['frames'].append(pointcloud)
            data['timestamps'].append(frame * 0.5)
        
        return data
    
    def save(self, output_dir):
        """保存所有数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"保存Dummy数据到: {output_dir}")
        print(f"{'='*60}\n")
        
        # 保存相机数据
        cam_data = self.generate_cameras()
        np.savez(
            output_dir / "cameras.npz",
            poses=np.array(cam_data['poses']),
            intrinsics=np.array(cam_data['intrinsics']),
            timestamps=np.array(cam_data['timestamps'])
        )
        
        # 保存文件名列表
        with open(output_dir / "image_filenames.txt", 'w') as f:
            for fname in cam_data['image_filenames']:
                f.write(fname + '\n')
        
        # 保存lidar数据
        lidar_data = self.generate_lidars()
        for i, (points, ts) in enumerate(zip(lidar_data['frames'], lidar_data['timestamps'])):
            np.save(output_dir / f"lidar_{i:04d}.npy", points)
        
        np.save(output_dir / "lidar_timestamps.npy", np.array(lidar_data['timestamps']))
        
        # 保存元信息
        meta = {
            'num_frames': self.num_frames,
            'num_cameras': self.num_cameras,
            'image_height': self.img_h,
            'image_width': self.img_w,
            'lidar_points_per_frame': [len(f) for f in lidar_data['frames']],
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✓ 相机数据: cameras.npz ({len(cam_data['poses'])} poses)")
        print(f"✓ Lidar数据: lidar_*.npy ({self.num_frames} 帧)")
        print(f"✓ 元数据: metadata.json")
        print(f"\n{'='*60}\n")
        
        return cam_data, lidar_data


def verify_data(data_dir):
    """验证生成的数据"""
    data_dir = Path(data_dir)
    
    print(f"\n{'='*60}")
    print("验证Dummy数据")
    print(f"{'='*60}\n")
    
    # 加载并检查
    cameras = np.load(data_dir / "cameras.npz")
    print(f"✓ 相机poses: {cameras['poses'].shape}")
    print(f"✓ 相机内参: {cameras['intrinsics'].shape}")
    print(f"✓ 时间戳: {cameras['timestamps'].shape}")
    
    # 检查第一帧
    print(f"\n第一个相机pose:")
    print(cameras['poses'][0])
    print(f"\n第一个相机内参 [fx, fy, cx, cy]:")
    print(cameras['intrinsics'][0])
    
    # Lidar
    lidar_files = sorted(data_dir.glob("lidar_*.npy"))
    print(f"\n✓ Lidar文件数: {len(lidar_files)}")
    if lidar_files:
        first_cloud = np.load(lidar_files[0])
        print(f"✓ 第一帧点云形状: {first_cloud.shape}")
        print(f"  - 点数: {len(first_cloud)}")
        print(f"  - 数据维度: {first_cloud.shape[1]} [x,y,z,intensity]")
        
        # 统计
        print(f"\n点云统计:")
        print(f"  X范围: [{first_cloud[:, 0].min():.1f}, {first_cloud[:, 0].max():.1f}]")
        print(f"  Y范围: [{first_cloud[:, 1].min():.1f}, {first_cloud[:, 1].max():.1f}]")
        print(f"  Z范围: [{first_cloud[:, 2].min():.1f}, {first_cloud[:, 2].max():.1f}]")
        print(f"  强度范围: [{first_cloud[:, 3].min():.1f}, {first_cloud[:, 3].max():.1f}]")
    
    # 元数据
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    print(f"\n元数据:")
    print(json.dumps(meta, indent=2))
    
    print(f"\n{'='*60}")
    print("✓ 数据验证通过!")
    print(f"{'='*60}\n")


def generate_usage_example(data_dir):
    """生成使用示例代码"""
    code = f"""
# ============ 如何使用这些Dummy数据 ============

import numpy as np
from pathlib import Path

# 1. 加载相机数据
data_dir = Path("{data_dir}")
cam_data = np.load(data_dir / "cameras.npz")

poses = cam_data['poses']          # (N, 4, 4) 相机pose矩阵
intrinsics = cam_data['intrinsics']  # (N, 4) [fx, fy, cx, cy]
timestamps = cam_data['timestamps']  # (N,) 时间戳

print(f"加载了 {{len(poses)}} 个相机pose")

# 2. 加载Lidar数据
lidar_files = sorted(data_dir.glob("lidar_*.npy"))
for i, lidar_file in enumerate(lidar_files):
    pointcloud = np.load(lidar_file)  # (M, 4) [x, y, z, intensity]
    print(f"帧 {{i}}: {{len(pointcloud)}} 个点")

# 3. 在训练循环中使用
for epoch in range(10):
    # 随机采样一个batch
    batch_size = 4
    indices = np.random.randint(0, len(poses), batch_size)
    
    batch_poses = poses[indices]
    batch_intrinsics = intrinsics[indices]
    
    # ... 你的模型前向传播 ...
    print(f"Epoch {{epoch}}, batch shape: {{batch_poses.shape}}")

# 4. 集成到NeuRAD
# 修改 nerfstudio/data/dataparsers/nuscenes_dataparser.py
# 在 _get_cameras() 中:
#   - 读取 {data_dir}/cameras.npz
#   - 转换为 Cameras 对象
# 在 _get_lidars() 中:
#   - 读取 {data_dir}/lidar_*.npy
#   - 转换为 Lidars 对象

print("\\n搞定! 数据已经准备好了")
"""
    
    print("\n" + "="*60)
    print("使用示例代码")
    print("="*60)
    print(code)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     NuScenes Dummy数据生成器 (纯NumPy版本)              ║
║     无依赖 | 快速 | 实用                                ║
║     "Talk is cheap. Show me the code." - Linus          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 生成数据
    output_dir = "/home/s0002322/Fisheye-Project/src/SplatAD-Ray-Tracing/neurad-studio/data/dummy_nuscenes_data"
    
    generator = MinimalDummyNuScenes(
        num_frames=5,
        num_cameras=6
    )
    
    cam_data, lidar_data = generator.save(output_dir)
    
    # 验证
    verify_data(output_dir)
    
    # 使用示例
    generate_usage_example(output_dir)
    
    print("\n下一步:")
    print("  1. 检查数据: ls -lh /home/claude/dummy_nuscenes_data/")
    print("  2. 用Python加载验证: python -c 'import numpy as np; d=np.load(...)'")
    print("  3. 集成到你的训练pipeline\n")


if __name__ == "__main__":
    main()

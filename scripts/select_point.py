import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  # 用于颜色映射

ceiling_height = 2.2 # 天花板高度
pcd_path = "/home/heron/SJTU/Codes/VIO/cross_modal_ws/src/global_loop/PCD/ref16.pcd"
def main():
    # 读取 PCD 文件 (Tensor格式)
    pcd = o3d.t.io.read_point_cloud(pcd_path, format='pcd')
    
    # 打印点云信息
    print(pcd)

    # 获取点的位置信息 (XYZ)
    pcd_xyz = pcd.point["positions"].numpy()
    print(f'pcd_xyz: {pcd_xyz}, shape: {pcd_xyz.shape}')
    
    # 获取强度信息并归一化到 [0, 1] 范围
    pcd_intensity = pcd.point["intensity"].numpy().ravel()  # 展平强度数组
    print(f'pcd_intensity: {pcd_intensity}, shape: {pcd_intensity.shape}')
    
    # 归一化强度值到 [0, 1]
    intensity_min = np.min(pcd_intensity)
    intensity_max = np.max(pcd_intensity)
    normalized_intensity = (pcd_intensity - intensity_min) / (intensity_max - intensity_min)
    print(f'Normalized intensity: {normalized_intensity}, shape: {normalized_intensity.shape}')

    # 使用 Jet 颜色映射将强度值转换为 RGB 颜色
    cmap = plt.get_cmap('jet')  # 可以选择 'jet', 'viridis', 'plasma' 等
    colors = cmap(normalized_intensity)[:, :3]  # 只取RGB三个通道
    print(f'colors: {colors}, shape: {colors.shape}')

    # 创建一个新的Legacy格式点云对象
    legacy_pcd = o3d.geometry.PointCloud()

    # 遍历所有点，将XYZ和颜色RGB分别存储在Legacy点云中
    for i in range(len(pcd_xyz)):
        if pcd_xyz[i][2] > ceiling_height:
            legacy_pcd.points.append(pcd_xyz[i])
            legacy_pcd.colors.append(colors[i])

    # 打印醒目的提示信息
    print("\033[1;32mPress [shift + left click] to select a point. And [shift + right click] to remove point.\033[0m")
    
    # 创建一个可视化窗口
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(legacy_pcd)  # 使用Legacy格式点云进行可视化
    vis.run()  # 用户在此窗口中选择点
    vis.destroy_window()

    # 获取选定点的索引
    picked_points = vis.get_picked_points()
    print("Picked points:", picked_points)

    # 打印选定点的坐标
    selected_points = []
    for idx in picked_points:
        point = np.asarray(legacy_pcd.points)[idx]
        selected_points.append(point)
        print("Point index:", idx, "Coordinates:", point)

    # 将选定点的坐标输出到文件
    with open("selected_points.xml", "w") as f:
        f.write(f'<arg name="point_num" value="{len(selected_points)}" />\n')
        for i, point in enumerate(selected_points):
            x = round(point[0], 1)
            y = round(point[1], 1)
            z = 0.7  # 默认值
            # f.write(f'<!-- {i} -->\n')
            f.write(f'<arg name="point{i}_x" value="{x}" />\n')
            f.write(f'<arg name="point{i}_y" value="{y}" />\n')
            f.write(f'<arg name="point{i}_z" value="{z}" />\n')

if __name__ == "__main__":
    main()

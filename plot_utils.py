from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray


class PlotUtils:
    @staticmethod
    def _set_xyz_lim(ax, x, y, z=None):
        # 设置长宽比相同
        ax.set_aspect("equal")

        # 确保x, y, z轴具有相同的刻度
        if z is None:
            max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max() / 2.0
        else:
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        max_range *= 1.1

        # 设置中心点
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5

        if z is not None:
            mid_z = (z.max() + z.min()) * 0.5

        # 设置x, y, z轴范围
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        if z is not None:
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ...

    @classmethod
    def plot_3d_coord(cls, coord):
        """
        plot the scatter in 3d
        :param coord: ndarray of shape(num_points, 3)
        """
        assert coord.ndim == 2 and coord.shape[1] == 3
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

        plt.figure(figsize=(10, 6))

        ax = plt.subplot(projection="3d")  # 创建一个三维的绘图工程
        cls._set_xyz_lim(ax, x, y, z)

        ax.set_title("3d_image_show")  # 设置本图名称
        ax.scatter(x[0], y[0], z[0], c="r")  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
        ax.scatter(x[1:], y[1:], z[1:], c="b")  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

        ax.set_xlabel("X")  # 设置x坐标轴
        ax.set_ylabel("Y")  # 设置y坐标轴
        ax.set_zlabel("Z")  # 设置z坐标轴

        plt.show()
        ...

    @classmethod
    def plot_2d_coord(
        cls,
        coords: ndarray,
        interval: float = 0.3,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        assert coords.ndim == 2 and coords.shape[1] in (2, 3)
        if coords.shape[1] == 3:
            coords = coords[:, :2]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        # 创建绘图窗口
        fig, ax = plt.subplots()
        cls._set_xyz_lim(ax, x_coords, y_coords)

        ax.grid(True)

        # 初始化点列表
        x_data = []
        y_data = []
        colors = []  # 用于存储每个点的颜色

        # 创建空的散点图
        scatter = ax.scatter([], [])

        # 更新绘图
        def update_points(x, y, color):
            x_data.append(x)
            y_data.append(y)
            colors.append(color)  # 添加新的点的颜色
            scatter.set_offsets(np.c_[x_data, y_data])
            scatter.set_color(colors)
            plt.draw()

        # 模拟每隔0.5秒更新一次点的位置
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.title(f"mic {i}")
            if i == 0:
                update_points(x, y, "red")  # 第一个点是红色
            else:
                update_points(x, y, "blue")  # 其他点是蓝色
            # 在图上标注点的顺序
            ax.text(x, y, str(i), fontsize=10, ha="right")
            plt.pause(interval)

        plt.title("2d mic coords")

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.show()
        ...

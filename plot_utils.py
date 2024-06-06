from matplotlib import pyplot as plt


class PlotUtils:
    @staticmethod
    def plot_3d_coord(coord):
        """
        plot the scatter in 3d
        :param coord: ndarray of shape(num_points, 3)
        """
        assert coord.ndim == 2 and coord.shape[1] == 3
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

        plt.figure(figsize=(15, 7))

        ax = plt.subplot(projection="3d")  # 创建一个三维的绘图工程
        ax.set_title("3d_image_show")  # 设置本图名称
        ax.scatter(x, y, z, c="b")  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

        ax.set_xlabel("X")  # 设置x坐标轴
        ax.set_ylabel("Y")  # 设置y坐标轴
        ax.set_zlabel("Z")  # 设置z坐标轴

        plt.show()
        ...

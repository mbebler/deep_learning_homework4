"""
"""
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        self.fig = fig
        self.axes = axes
        self.frames = []

    def process(self, sample: dict, debug_info: dict | None = None):
        fig, axes = self.fig, self.axes

        for ax in axes:
            ax.clear()

        axes[0].imshow(sample["image_raw"])

        if debug_info:
            axes[1].plot(debug_info["waypoints"][:, 0], debug_info["waypoints"][:, 1], "b-o")
            axes[1].set_title(f"Steer: {debug_info['steer']:.2f} Speed: {debug_info['speed']:.2f}")
            axes[1].set_xlim(-10, 10)
            axes[1].set_ylim(-5, 15)

        s, (width, height) = fig.canvas.print_to_buffer()
        viz = np.frombuffer(s, np.uint8).reshape((height, width, 4))[:, :, :3]

        self.frames.append(viz)


def save_video(images, filename="video.mp4"):
    import imageio

    with imageio.get_writer(filename, mode="I") as writer:
        for img in images:
            writer.append_data(img)

    print(f"{len(images)} frames saved to {filename}")

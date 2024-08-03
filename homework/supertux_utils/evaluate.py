import numpy as np
import torch

from ..datasets.road_transforms import EgoTrackProcessor
from ..datasets.road_utils import Track
from .race import rollout
from .visualizations import Visualizer


class BasePlanner:
    """
    Base class for learning-based planners.
    """
    # list of allowed information to be returned
    ALLOWED_INFORMATION = []

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
    ):
        self.model = model
        self.model.to(device).eval()

        self.debug_info = {}

    @torch.inference_mode()
    def act(self, batch: dict) -> dict:
        allowed_info = {k: batch.get(k) for k in self.ALLOWED_INFORMATION}
        pred = self.model(**allowed_info)

        speed = np.linalg.norm(batch["velocity"].squeeze(0).cpu().numpy())
        steer, acceleration, brake = self.get_action(pred, speed)

        return {
            "steer": steer,
            "acceleration": acceleration,
            "brake": brake
        }

    def get_action(
        self,
        waypoints: torch.Tensor,
        speed: torch.Tensor,
        target_speed: float = 5.0,
        idx: int = 2,
        p_gain: float = 10.0,
        constant_acceleration: float = 0.2,
    ) -> tuple[float, float, bool]:
        """
        Args:
            waypoints (torch.Tensor): waypoint predictions for a single sample (n, 2) or (1, n, 2)

        Returns:
            steer (float) from -1 to 1
            acceleration (float) from 0 to 1
            brake (bool) whether to brake
        """
        # make sure waypoints are (n, 2)
        waypoints = waypoints.squeeze(0).cpu().numpy()

        # steering angle is proportional to the angle of the target waypoint
        angle = np.arctan2(waypoints[idx, 0], waypoints[idx, 1])
        steer = p_gain * angle

        # very simple speed control
        acceleration = constant_acceleration if target_speed > speed else 0.0
        brake = False

        # NOTE: you can modify use this and the visualizer to debug your model
        self.debug_info.update({"waypoints": waypoints, "steer": steer, "speed": speed})

        # clip to valid range
        steer = float(np.clip(steer, -1, 1))
        acceleration = float(np.clip(acceleration, 0, 1))

        return steer, acceleration, brake


class TrackPlanner(BasePlanner):
    ALLOWED_INFORMATION = ["bev_track_left", "bev_track_right"]


class ImagePlanner(BasePlanner):
    ALLOWED_INFORMATION = ["image"]


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str | None = None,
        visualizer: Visualizer | None = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_type = model.__class__.__name__
        model_to_planner = {
            "MLPPlanner": TrackPlanner,
            "TransformerPlanner": TrackPlanner,
            "CNNPlanner": ImagePlanner,
        }

        if model_type not in model_to_planner:
            raise ValueError(f"Model {model_type} not supported")

        self.planner = model_to_planner[model_type](model, self.device)
        self.visualizer = visualizer

        # lazily intialize the track later
        self.track = None
        self.track_transform = None

    @torch.inference_mode()
    def step(self, state, track, render_data):
        sample = {
            "location": np.float32(state.karts[0].location),
            "front": np.float32(state.karts[0].front),
            "velocity": np.float32(state.karts[0].velocity),
            "distance_down_track": float(state.karts[0].distance_down_track),
            "image_raw": np.uint8(render_data[0].image),
        }

        track_info = self.track_transform.from_frame(**sample)
        sample.update(track_info)
        sample['image'] = np.float32(sample['image_raw']).transpose(2, 0, 1) / 255.0

        # turn all numpy into batch size=1 tensors
        batch = torch.utils.data.default_collate([sample])
        # hack: torch upcasts it on some machines
        batch['distance_down_track'] = batch['distance_down_track'].float()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        action = self.planner.act(batch)

        # optionally save/visualize frame info
        if self.visualizer is not None:
            self.visualizer.process(sample, self.planner.debug_info)

        return action

    def evaluate(
        self,
        track_name: str = "lighthouse",
        max_steps: int = 100,
        frame_skip: int = 4,
        disable_progress: bool = False,
    ):
        max_distance = 0.0
        total_track_distance = float("inf")

        with rollout(
            callback=self.step,
            track_name=track_name,
            max_steps=max_steps,
            frame_skip=frame_skip,
            disable_progress=disable_progress,
        ) as rollout_loop:
            for i, payload in enumerate(rollout_loop):
                state = payload["state"]
                render_data = payload["render_data"]
                pystk_track = payload["track"]

                # update how far the kart has gone
                max_distance = max(max_distance, state.karts[0].distance_down_track)

                # only set track once on first frame
                if i == 0:
                    self.track = Track(
                        path_distance=pystk_track.path_distance,
                        path_nodes=pystk_track.path_nodes,
                        path_width=pystk_track.path_width,
                    )
                    self.track_transform = EgoTrackProcessor(self.track)

                    # set total distance now that we have the track
                    total_track_distance = pystk_track.path_distance[-1][0]

        return max_distance, total_track_distance

from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    class MLP_Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin1 = nn.Linear(in_channels, out_channels)
            self.relu = nn.ReLU()
            self.lin2 = nn.Linear(out_channels, out_channels)
            self.batch1 = nn.BatchNorm1d(out_channels)
            self.batch2 = nn.BatchNorm1d(out_channels)
            if in_channels != out_channels:
                self.skip = nn.Linear(in_channels, out_channels)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            x1 = self.relu(self.batch1(self.lin1(x)))
            # print(x1.shape)
            x2 = self.relu(self.batch2(self.lin2(x1)))
            # print(x2.shape)
            return x2 + self.skip(x)

    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        layers = nn.ModuleList()
        c1 = 4 * self.n_track  # when we flatten, we pass in 2 left and 2 y values
        for _ in range(3):  # run through the block three times
            c2 = 2 * c1
            layers.append(self.MLP_Block(c1, c2))
            c1 = c2

        # final layer for the output - one output for everything
        layers.append(nn.Linear(c2, n_waypoints * 2))
        self.model = nn.Sequential(*layers)

    def forward(
            self,
            bev_track_left: torch.Tensor,
            bev_track_right: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints (b, n_waypoints, 2)
        from the left (b, n_track) and right boundaries (b, n_track) of the track.

        During test time, your model will be called with
        model(bev_track_left=foo, bev_track_right=bar) with no other keyword arguments,
        so make sure you are not assuming any additional inputs here.

        Args:
            bev_track_left (torch.Tensor): shape (b, n_track, 2)
            bev_track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = torch.cat((bev_track_left, bev_track_right), dim=1).flatten(start_dim=1)  # concat to feed in
        y = self.model(x)
        y2 = y.view(x.shape[0], self.n_waypoints, 2)  # reconfigure into the correct size
        return y2


class TransformerPlanner(nn.Module):
    class TransformerLayer(torch.nn.Module):
        def __init__(self, embed_dim, heads):
            super().__init__()
            # first we have the self attention
            self.self_att = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
            # then, we add an MLP
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, 2 * embed_dim),
                torch.nn.BatchNorm1d(embed_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * embed_dim, embed_dim),
                torch.nn.BatchNorm1d(embed_dim),
                torch.nn.ReLU(),
            )
            # then 2 normalizations
            self.norm1 = torch.nn.LayerNorm(embed_dim)
            self.norm2 = torch.nn.LayerNorm(embed_dim)

        def forward(self, x):
            x_norm = self.norm1(
                x)  # could add in a cross attention layer with encoder/decoder architecture but not absolutely necessary
            x = x + self.self_att(x_norm, x_norm, x_norm)[
                0]  # attention outputs layers + weights, but we only want the layer solution
            x_norm2 = self.norm2(x)
            x = x + self.mlp(x_norm2)
            return x

    def __init__(
            self,
            n_track: int = 10,
            n_waypoints: int = 3,
            d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # using the basic model from week 6 lectures

        layers = nn.ModuleList()
        self.layers = torch.nn.Sequential(
            *[self.TransformerLayer(n_track * 4, 4) for _ in range(d_model)],
            *[self.TransformerLayer(n_track * 4, 4) for _ in range(d_model)],
        )
        self.layers.append(nn.Linear(4 * n_track, n_waypoints * 2))

    def forward(
            self,
            bev_track_left: torch.Tensor,
            bev_track_right: torch.Tensor,
            **kwargs,):

            """
            Predicts waypoints (b, n_waypoints, 2)
            from the left (b, n_track) and right boundaries (b, n_track) of the track.

            During test time, your model will be called with
            model(bev_track_left=foo, bev_track_right=bar) with no other keyword arguments,
            so make sure you are not assuming any additional inputs here.

            Args:
                bev_track_left (torch.Tensor): shape (b, n_track, 2)
                bev_track_right (torch.Tensor): shape (b, n_track, 2)

            Returns:
                torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
            """
            x=torch.cat((bev_track_left, bev_track_right), dim=1).flatten(start_dim=1)  # concat to feed in
            y = self.layers(x)
            y2 = y.view(y.shape[0], self.n_waypoints, 2)  # reconfigure into the correct size

            return y2
class CNNPlanner(torch.nn.Module):
    class Block(nn.Module):
        def __init__(self, in_chan=3, out_chan=3):
            super().__init__()
            self.con1 = torch.nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1)
            self.relu = torch.nn.ReLU()
            self.norm = torch.nn.BatchNorm2d(out_chan)
            self.con2 = torch.nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=1, stride=1)
            if in_chan == out_chan:
                self.skip = torch.nn.Identity()
            else:
                self.skip = torch.nn.Conv2d(in_chan, out_chan, 1)

        def forward(self, x):
            x1 = self.norm(self.relu(self.con1(x)))
            # print(x1.shape)
            x2 = self.relu(self.con2(x1))
            # print(x2.shape)
            return x2 + self.skip(x)

    def __init__(
            self,
            n_waypoints: int = 3
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        c1 = 8
        # first layer
        layers = [
            torch.nn.Conv2d(3, out_channels=c1, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
        ]
        # for the blocks
        for _ in range(6):  # 3 total layers
            c2 = 2 * c1
            layers.append(self.Block(c1, c2))
            c1 = c2
        #layers.append(torch.nn.Dropout(p=0.5))
        self.model = torch.nn.Sequential(*layers)
        # for the last layer
        # self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        final_feat = (2 ** 6) * 8
        self.end = torch.nn.Linear(in_features=final_feat, out_features=n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """

        out = self.model(image)
        out = torch.mean(out, dim=(2, 3))
        out = self.end(out)
        out = out.view(image.shape[0], self.n_waypoints, 2)

        return out

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
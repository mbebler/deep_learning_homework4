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
        c1 = 4*self.n_track #when we flatten, we pass in 2 left and 2 y values
        for _ in range(3):  # run through the block three times
            c2 = 2 * c1
            layers.append(self.MLP_Block(c1, c2))
            c1 = c2

        # final layer for the output - one output for everything
        layers.append(nn.Linear(c2, n_waypoints*2))
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
        x = torch.cat((bev_track_left, bev_track_right), dim=1).flatten(start_dim=1) #concat to feed in
        y = self.model(x)
        y2 = y.view(x.shape[0], self.n_waypoints, 2) # reconfigure into the correct size
        return y2



class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        #using the basic model from week 6 lectures
        self.query_embed = nn.Embedding(n_waypoints, d_model) #self.enc
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model*n_track, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_waypoints)
        )

    def forward(
        self,
        bev_track_left: torch.Tensor,
        bev_track_right: torch.Tensor,
        **kwargs,
    ):
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
        return self.net(self.query_embed(bev_track_left + bev_track_right))


class CNNPlanner(torch.nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_chan, out_chan):
            super().__init__()
            self.con1 = torch.nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1,
                                        bias=False)
            self.relu = torch.nn.ReLU()
            self.norm1 = torch.nn.BatchNorm2d(out_chan)
            self.con2 = torch.nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=3, stride=1, padding=1)
            self.norm2 = torch.nn.BatchNorm2d(out_chan)

        def forward(self, x):
            x1 = self.relu(self.norm1((self.con1(x))))
            #print(x1.shape)
            x2 = self.relu(self.norm2((self.con2(x1))))
            return x2

    def __init__(
        self,
        n_waypoints: int = 3,
        channel_block= [16, 32, 64]
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        self.down_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1 = n_waypoints
        for i in channel_block:
            # we first run it through the down convolution blocks
            self.down_conv.append(self.ConvBlock(in_chan=c1, out_chan=i))
            c1 = i


        # now base layer
        self.base = self.ConvBlock(channel_block[-1], channel_block[-1]*2)

        for j in reversed(channel_block):
            # now to go back up
            self.up_conv.append(nn.ConvTranspose2d(j * 2, j, kernel_size=2, stride=2))
            self.up_conv.append(self.ConvBlock(in_chan=j * 2, out_chan=j))

        # now to create the outputs
        self.logit_con = torch.nn.Conv2d(in_channels=j, out_channels=self.n_waypoints, kernel_size=1)


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        skip_connections = []
        for down in self.down_conv:  # the down cov path
            x = down(image)
            skip_connections.append(x)
            x = self.pool(x)
            # print(skip_connections)

        x = self.base(x)  # the base path
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.up_conv), 2):  # the up convolution
            x = self.up_conv[idx](x)
            # print(skip_connections[idx // 2])
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_conv[idx + 1](concat_skip)

        # the final output layers
        return self.logit_con(x)


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

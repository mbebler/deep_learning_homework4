"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import torch.utils.tensorboard as tb

from .models import MLPPlanner, TransformerPlanner, save_model, calculate_model_size_mb
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data

'''
The previous homework should be a good reference, but feel free to modify different parts of the training code depending on how you want to perform experiments.

Recall that a training pipeline includes:
* Creating an optimizer
* Creating a model, loss, metrics (task dependent)
* Loading the data (task dependent)
* Running the optimizer for several epochs
* Logging + saving your model (use the provided `save_model`)
'''

def train(
        exp_dir: str = "logs",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 2024,
        vers: bool = True
):
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # loading the data
    train_data = load_data("deep_learning_homework4/road_data/train", shuffle=True,
                           batch_size=batch_size)
    val_data = load_data("deep_learning_homework4/road_data/val", shuffle=False)

    # load the model
    if vers:
        net = MLPPlanner()
    else:
        net = TransformerPlanner()
    net.cuda()
    net.train()

    # create the loss function and optimizer
    loss_func = torch.nn.L1Loss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()
    global_step = 0

    # now to train the model
    for epoch in range(num_epoch):
      # print(epoch)

      net.train()
      for data in train_data:
        # input the data
        left_bev = data["bev_track_left"].cuda()
        right_bev = data["bev_track_right"].cuda()
        way = data["waypoints"].cuda()
        way_mask = data["waypoints_mask"].cuda()
        # find the model
        output = net(bev_track_left = left_bev, bev_track_right = right_bev)
        # calculate the loss, add to the accuracy tracker
        new_loss = loss_func(output, way)
        logger.add_scalar("train/loss", new_loss.item(), global_step)
        # now update the model
        optim.zero_grad()
        loss_func.backward()
        optim.step()
        # add the individual loss to the logger
        train_metrics.add(output, way, way_mask)

        global_step += 1

      # now we add the overall accuracy to the logger
      train_metrics2 = train_metrics.compute()
      logger.add_scalar("train/accuracy-long", np.mean(train_metrics2["longitudinal_error"]), global_step=epoch)
      logger.add_scalar("train/accuracy-lat", np.mean(train_metrics2['lateral_error']), global_step=epoch)

      # now we want to look at the validation data
      net.eval()
      val_accuracy = []
      for data in val_data:
        # input the data
        left_bev = data["bev_track_left"].cuda()
        right_bev = data["bev_track_right"].cuda()
        way = data["waypoints"].cuda()
        way_mask = data["waypoints_mask"].cuda()
        # find the model
        with torch.inference_mode():
          output = net(bev_track_left = left_bev, bev_track_right = right_bev)
          # print((net.predict(x = data) == label).float().mean())
          val_metrics.add(output, way, way_mask)


        #log the mean val accuracy in logger
        val_metrics2 = val_metrics.compute()
        logger.add_scalar("val/accuracy-long", np.mean(val_metrics2["longitudinal_error"]), global_step=epoch)
        logger.add_scalar("val/accuracy-lat", np.mean(val_metrics2['lateral_error']), global_step=epoch)

      logger.flush()

        # now we save models every so often
      if epoch % 10 == 0:
        save_model(net)
        print(calculate_model_size_mb(net))
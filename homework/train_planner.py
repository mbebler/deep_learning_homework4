"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import numpy as np
import torch

import torch.utils.tensorboard as tb

from .models import MLPPlanner, save_model, calculate_model_size_mb
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

def train_MLP(
        exp_dir: str = "logs",
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        seed: int = 2024,
):
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # loading the data
    train_data = load_data("DL_hmwk_4/road_data/train", shuffle=True,
                           batch_size=batch_size)
    val_data = load_data("DL_hmwk_4/road_data/val", shuffle=False)

    # load the model
    net = MLPPlanner()
    net.cuda()
    net.train()

    # create the loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    global_step = 0

    # now to train the model
    for epoch in range(num_epoch):
      # print(epoch)
      train_accuracy = []
      net.train()
      for DATA in train_data:
        # input the data

        # find the model
        output = net(data)
        # calculate the loss, add to the accuracy tracker

        # now update the model
        optim.zero_grad()
        loss.backward()
        optim.step()
        # add the individual loss to the logger

        global_step += 1

      # now we add the overall accuracy to the logger


      # now we want to look at the validation data
      net.eval()
      val_accuracy = []
      for DATA in val_data:

        with torch.inference_mode():
          output = net(data)
          # print((net.predict(x = data) == label).float().mean())
        #log the val accuracy

        #log the mean val accuracy in logger
        logger.add_scalar("val/accuracy", np.mean(val_accuracy), global_step=epoch)

      logger.flush()

        # now we save models every so often
      if epoch % 10 == 0:
        save_model(net)
        print(calculate_model_size_mb(net))
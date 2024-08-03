# Homework 4

In this homework, we will learn to drive with Transformers and convolutional networks!

You will need to use a GPU or [Google Colab](https://colab.research.google.com/) to train your models.

## Setup + Starter Code

In this assignment, we'll be working with the [SuperTuxKart Road Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/road_data.zip).
**NOTE:** Make sure to re-download the dataset! We've added some additional metadata needed for this homework.

You can download the dataset from your terminal by running the following command from the main directory:
```bash
curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/road_data.zip -o ./road_data.zip && unzip -qo road_data.zip
```

Verify that your working directory looks like this: 
```
bundle.py
grader
homework
road_data
```
You will run all scripts from inside this main directory.

In the `homework` directory, you'll find the following:
- `models.py` - where you will implement various models
- `metrics.py` - metrics to evaluate your models
- `datasets` - contains dataset loading and transformation functions
- `supertux_utils` - game wrapper + visualization (optional)

## Training

As in the previous homework, you will implement the training code from scratch!
This might seem cumbersome modifying the same code repeatedly, but this will help understand the engineering behind writing model/data agnostic training pipelines.

Recall that a training pipeline includes:
* Creating an optimizer
* Creating a model, loss, metrics
* Loading the data
* Running the optimizer for several epochs
* Logging + saving your model (use the provided `save_model`)

### Grader Instructions

You can grade your implementation after any part of the homework by running the following command from the main directory:
- `python3 -m grader homework -v` for medium verbosity
- `python3 -m grader homework -vv` to include print statements

## Part 1a: MLP Planner (35 points)

In this part, we will implement a MLP to learn how to drive!
Rather than learning from images directly, we will predict the desired trajectory of the vehicle from the ground truth lane boundaries (similar to what we were trying to segment in Homework 3).
After we have these the desired future trajectory (waypoints), we can use a simple controller to follow the waypoints and drive the vehicle in PySuperTuxKart.

### [Road Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/road_data.zip)

To learn this model, we'll need the following data:
* `left_lane_boundaries` - `(n_track, 2)` sequence of left lane boundaries
* `right_lane_boundaries` - `(n_track, 2)` sequence of right lane boundaries
* `waypoints` - `(n_waypoints, 2)` vehicle positions at the next `n_waypoints` time-steps
* `waypoints_mask` - `(n_waypoints,)` mask indicating whether the time-step is valid

To supervise our model, we'll need labels for where we should navigate to.
For this homework, we'll try to predict the recorded autopilot's future trajectory.
For a given time-step `t`, we can process the future positions at `t+1, t+2, t+3` into waypoints by simply transforming them into the coordinate system at time `t`.
Since the sequences are finite-length, we'll use the `waypoints_mask` to indicate which points are valid (only the last couple of timesteps have invalid points).

For our inputs, we'll use the closest left and right boundaries of the track, projected into the current timestamp's coordinate frame.
In Part 1a, we won't need to use the images, so make sure to specify the `road_transform` accordingly.

Relevant code:
* `datasets/road_dataset.py:RoadDataset.get_transform`
* `datasets/road_transforms.py:EgoTrackProcessor`

The data processing functions are already implemented, but feel free to add custom transformations for data augmentation.

### Model

Implement the `MLPPlanner` model in `models.py`.
Your `forward` function receives a `(B, n_track, 2)` tensor of left lane boundaries and a `(B, n_track, 2)` tensor of right lane boundaries and should return a `(B, n_waypoints, 2)` tensor of predicted vehicle positions at the next `n_waypoints` time-steps.

Find a suitable loss function to train your model.
The number of points `n_track=10` and `n_waypoints=3` is guaranteed to be fixed for all parts in the homework.

### Evaluation

For all parts in this homework, we will evaluate the predicted trajectory with two offline metrics.
Longitudinal error (absolute difference in the forward direction) is a good proxy for how well the model can predict the speed of the vehicle, while lateral error (absolute difference in the left/right direction) is a good proxy for how well the model can predict the steering of the vehicle.

Once your model is able to predict the trajectory well, we can actually run the model in PySuperTuxKart to see how well it drives!

This portion requires installing PySuperTuxKart, which requires some careful setup.
```bash
pip install PySuperTuxKartData
pip install PySuperTuxKart --index-url=https://www.cs.utexas.edu/~bzhou/dl_class/pystk

# if that doesn't work and you REALLY want to see the driving, you can try building for source
# this is only recommended if you have previous experience with buildng python/c++ packages
# https://github.com/bradyz/pystk
```

If you can't get PySuperTuxKart running, don't worry since we'll be able to evaluate your model when you submit.
The offline metrics are a strong proxy for how well the model will perform when driving in game.

If you want to visualize the driving, see the following files in `supertux_utils` module:
* `race.py` - the interface to PySuperTuxKart (no need to fully understand this)
* `evaluate.py` - contains logic on how the model's predictions are used to drive + what data is logged
* `visualizations.py` - uses matplotlib to visualize the driving + some utility to save a video (requires `imageio` to be installed)

then you'll need to create your own visualization script, here's a simple example:
```python
from .supertux_utils.evaluate import Evaluator
from .models import load_model
from .supertux_utils.visualizations import Visualizer, save_video

model = load_model("mlp_planner", with_weights=True)

# run the model on the track
visualizer = Visualizer()

# set visualizer to None if you don't want to visualize
evaluator = Evaluator(model, visualizer=visualizer)
evaluator.evaluate(track_name='lighthouse', max_steps=100)

# list of images (numpy array)
frames = visualizer.frames

# visualize however you like (plt.imshow, make a video, etc)
save_video(frames, "video.mp4")
```

### Part 1b: Transformer Planner (35 points)

We'll build a similar model to Part 1a, but this time we'll use a Transformer.
One way to do this is by using a set of `n_waypoints` learned query embeddings to attend over the set of points in lane boundaries (cross attention using the lane boundaries as keys and values and the waypoint embeddings as queries).

Compared to the MLP model, there are many more ways to design this model!
Training the transformer will likely require more tuning, so make sure to optimize your training pipeline to allow for faster experimentation.

### Relevant Operations
- [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [torch.nn.TransformerDecoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html)
- [torch.nn.TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)

## Part 2: CNN Planner (30 points)

One major limitation of the previous models is that they require the ground truth lane boundaries as input.
In the previous homework, we trained a model to predict these in image space, but reprojecting the lane boundaries from image space to the vehicle's coordinate frame is non-trivial as small depth errors are magnified through the re-projection process.
Rather than going through segmentation and depth estimation, we can learn to predict the lane boundaries in the vehicle's coordinate frame directly from the image!

Implement the `CNNPlanner` model in `models.py`.
Your `forward` function receives a `(B, 3, 96, 128)` image tensor as input and should return a `(B, n_waypoints, 2)` tensor of predicted vehicle positions at the next `n_waypoints` time-steps.
The previous homeworks will be strong baselines for this model.

## Submission

Create a submission bundle (max size **60MB**) using:
```bash
python3 bundle.py homework [YOUR UT ID]
```

Please double-check that your zip file was properly created by grading it again.
```bash
python3 -m grader [YOUR UT ID].zip
```
After verifying that the zip file grades successfully, you can submit it on Canvas.

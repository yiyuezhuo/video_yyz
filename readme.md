# Yet another video classification framework

I wrote this framework for a steel mill project. I don't spend one second searching if a similar project does exist (I guess there it is!), just an exercise project.

## Usage

### Style

While `train.py` provides a CLI style interface, I found this style, also appearing in `torchvision`, is hard to understand and maintain. So I force users to add a function, called "frozen config", in `frozen_xyz` and use its name in `train.py`.

For example, you can see following content in `frozen_models.py`:

```python
def cnn_lstm_1():
    return models.cnn_lstm(num_classes=3, pretrained=True)
```

You can use `--model cnn_lstm_1` to point to it in `train.py`.

On the other hand, I still feel very annoyed to edit a pure text CLI command. So I suggest to use "frozen" commands like following:

```
python -m video_yyz.exps.test_word_bag_1
```

Here's the content of `test_word_bag_1`:

```python
import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_rgb', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_word_bag',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_1',
    '--num-epoch', "30",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train
```

While some weird hack is used to call a CLI script by importing, the further wrapping is not used to keep its simplicity.

So you can edit a script on your local, push it to the server, and type just a script name, rather than copy & paste something like `python train.py --a b --c d --e g ....`.

To reduce loading time, someone may use `ffmpeg` to split video into frames in advance. But I found that it may not be that effective, compared to `torchvision`'s method (decode frame from video dynamically). See following table:

```
resolution  load    speedup
1x          mp4     1.00
1x          jpg     1.01
0.5x        mp4     2.00
0.5x        jpg     4.09
0.25x       mp4     5.80
0.25x       jpg     12.48
```

Maybe surprisingly, if no resolution reduction is applied, even no speed up can be observed from data. It's due to that loading frames from image bottleneck on IO, and decoding frames from video bottleneck on CPU.

The speed up is significant when resolution reduction is applied. But obviously, the total time cost is small enough hence it doesn't matter to use decoding (in fact, if `num_workers != 0`, the cost is literally 0).

### Setup

#### Dataset file structure

Assumed file structure:

```
SVD
    video_sample
        dry
            40_31.mp4
            ...
        melt
            ...
        normal
            ...
```

`SVD` is an environment variable to denote dataset root. Use soft link to prevent  moving too many files.

#### Resize video to speed up loading

```shell
python video_yyz.fast resize_free
```

This script will resize video using ffmpeg and save them into `SVD/video_sample_free` folder.

(A split cache folder containing `*.jpg` is created as well, you can use `dataset.VideoDatasetFast` to leverage the "fast" version, though it's not fast enough to let me use it. So you can prevent splitting by commenting out `split_video` to save some disk space).

Add the `video_yyz` package into your `PYTHONPATH`, which is best practice when the package is still actively developed. Don't bother to `setup.py` to add it to `site-packages`.  

#### Install TensorBoard

Install TensorBoard. It's *not* optional, since it's such amazing.

### Train

Set environment variables `TRAIN_DATA`, `TEST_DATA` like:

```shell
export TRAIN_DATA=$SVD/video_sample_free
export TEST_DATA=$SVD/video_sample_free
```

Most of scripts in `exps` denote a "experiment", you can run one by something like:

```shell
python -m video_yyz.exps.test_word_bag_1
```

By using this style, you can call `%debug` using IPython to debug. like:

```
ipython -i -m video_yyz.exps.test_optical_3_resume_1
```

Checkpoints and TensorBoard logs will be stored in the current directory with default settings. Start `TensorBoard` using

```shell
tensorboard --logdir runs --port 8965
```

Then you can access it by `http://your_host:8965` from remote. 

## Results

`notebook_utils` provide some helpers to evaluate in a notebook. Such as following ensemble results:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>combination</th>
      <th>acc</th>
      <th>acc_adjusted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>resnet18_flat_L5</td>
      <td>0.855263</td>
      <td>0.776316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>cnn_lstm_1</td>
      <td>0.868421</td>
      <td>0.828947</td>
    </tr>
    <tr>
      <th>9</th>
      <td>r2plus1d_18_1, resnet18_flat_L5</td>
      <td>0.868421</td>
      <td>0.894737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>r2plus1d_18_1</td>
      <td>0.894737</td>
      <td>0.881579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>resnet18_word_bag, cnn_lstm_1</td>
      <td>0.894737</td>
      <td>0.855263</td>
    </tr>
    <tr>
      <th>13</th>
      <td>cnn_lstm_1, r2plus1d_18_1, resnet18_flat_L5</td>
      <td>0.894737</td>
      <td>0.868421</td>
    </tr>
    <tr>
      <th>0</th>
      <td>resnet18_word_bag</td>
      <td>0.907895</td>
      <td>0.842105</td>
    </tr>
    <tr>
      <th>5</th>
      <td>resnet18_word_bag, r2plus1d_18_1</td>
      <td>0.907895</td>
      <td>0.947368</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cnn_lstm_1, r2plus1d_18_1</td>
      <td>0.907895</td>
      <td>0.881579</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cnn_lstm_1, resnet18_flat_L5</td>
      <td>0.907895</td>
      <td>0.802632</td>
    </tr>
    <tr>
      <th>10</th>
      <td>resnet18_word_bag, cnn_lstm_1, r2plus1d_18_1</td>
      <td>0.907895</td>
      <td>0.881579</td>
    </tr>
    <tr>
      <th>11</th>
      <td>resnet18_word_bag, cnn_lstm_1, resnet18_flat_L5</td>
      <td>0.907895</td>
      <td>0.828947</td>
    </tr>
    <tr>
      <th>12</th>
      <td>resnet18_word_bag, r2plus1d_18_1, resnet18_flat_L5</td>
      <td>0.907895</td>
      <td>0.894737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>resnet18_word_bag, resnet18_flat_L5</td>
      <td>0.921053</td>
      <td>0.842105</td>
    </tr>
    <tr>
      <th>14</th>
      <td>resnet18_word_bag, cnn_lstm_1, r2plus1d_18_1, resnet18_flat_L5</td>
      <td>0.921053</td>
      <td>0.881579</td>
    </tr>
  </tbody>
</table>

<!-- Export this table using `print(df_sorted.to_html())`), greet formating tool pandas! (pandas: ??? -->

Two streams model itself is an ensemble including RGB(`resnet18_word_bag`) and L=5 optical flow model (`resnet18_flat_L5`). `cnn_lstm_1` is a CNN+LSTM example. `r2plus1d_18_1` is a 3d-CNN implementation brought from `torchvision`.

The result is from an average of 100 uniform clip samples prediction results from 10s video. A 75% "acc (frame)" can make up a 85% "acc (video)" as shown in [notebook](https://gist.github.com/yiyuezhuo/29d227880473183a090dc77022706fe8). "acc (adjusted)" is obtained by reassigning some low belief "normal" result as "abnormal", which prefer more "abnormal" prediction but usually make acc lower.

More improvement may be achieved by ensemble with transforms `LeftCrop`, `RightCrop` other than just `CenterCrop`.

It is worth noting that optical-flow models take longer time to train to achieve comparable performance compared to other (you can see some `resume` scripts for optical flow to increase training time). And `L=5` is not significantly greater than `L=1`. Just use TensorBoard to track these phenomenons.

`markov.py` can be used to predict at any time (any model running time) using a Markov Process.

<img src="https://pbs.twimg.com/media/EQEr3ZiUcAAIVhB?format=png&name=small">
<img src="https://pbs.twimg.com/media/EQEr3ZjUwAAsp8D?format=png&name=small">



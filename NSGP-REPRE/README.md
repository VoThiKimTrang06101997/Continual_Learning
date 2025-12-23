# Demystifying Catastrophic Forgetting in Two-Stage Incremental Object Detector

Hi, here is a brief instruction on how to run this coarsely cleaned code.

Our implementation is based on MMDetection 3.3.0. You can simply replace the corresponding files in MMDetection, and everything should work. However, we haven't tested the code after removing redundant experimental components, so we can't guarantee full functionality. Feel free to open an issue if you encounter any problems.

We have modified the runner and a few optimizers to implement Null Space Gradient Projection. You can find these changes in `mmdet/engine`.

For ROI Replay, we modified several models, including the Faster R-CNN detector, ROI head, and bbox head. These can be found in `mmdet/models`. Most of the modified files are marked with "roi\_replay" or "task" in their names.

To support easy dataset splitting for simulating sequential tasks in Incremental Object Detection (IOD), we also modified the dataset code. These can be found in `mmdet/datasets`. You can easily split the dataset into as many tasks as you need and control the size of each task by adjusting the dataset config. Examples can be found in the `cl_faster_rcnn_configs` directory, along with other configs such as 5-5 training setups and more.

We will release the cleaned-up version of our code as soon as possible. Sorry for the inconvenience.



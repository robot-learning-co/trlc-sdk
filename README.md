<h1 align="center">Build intelligent robots 🤖</h1>

[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.robot-learning.co/quickstart)
[![Twitter Follow](https://img.shields.io/twitter/follow/Jannik?style=social)](https://x.com/JannikGrothusen)

The goal of this API is to make AI-powered robotic automation accessible to anyone.

## Installation
```
git clone https://github.com/robot-learning-co/trlc-sdk.git && cd trlc-sdk
conda create -n trlc-sdk python=3.10 && conda activate trlc-sdk
pip install -e .
```

## Perception
Currently the perception API provides two endpoints:

1. **Grounded Segmentation** using [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) and
2. **6D Object Pose Estimation** using [FoundationPose](https://github.com/NVlabs/FoundationPose)

by combining these two we can extract the pose of any object from a RGB-D image:

<picture>
  <img src="./static/pose_estimation.png"  width="full">
</picture>

You can find examples for these endpoints and their combination in [examples/](examples/).

### Creating your own 3D object mesh
Follow the [documentation](https://docs.robot-learning.co/examples/pose_estimation#creating-a-3d-object-mesh) to create a 3D mesh from any object or CAD file:
<picture>
  <img src="./static/ar_capture.png"  width="full">
</picture>
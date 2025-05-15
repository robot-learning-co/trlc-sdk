## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

# from https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "out")
# Ensure output directory exists
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

RecordStream = False

# Streaming loop
try:
    print("Warming up camera for 30 frames...")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("Press Enter to capture one frame...")
    input()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    if not aligned_depth_frame or not color_frame:
        print("Could not capture frames. Exiting.")
    else:
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        timestamp = int(round(time.time() * 1000))

        with open(os.path.join(out_dir, "cam_K.txt"), "w") as f:
            f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
            f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
            f.write(f"{0.0} {0.0} {1.0}\n")

        cv2.imwrite(os.path.join(out_dir, f"rgb.png"), color_image)
        cv2.imwrite(os.path.join(out_dir, f"depth.png"), depth_image)

        print("Frames saved.")

finally:
    pipeline.stop()
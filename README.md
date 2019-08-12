# Introduction
   Demo to run PoseNet inference on Movidius NCS using OpenVINO.

   Look at `run_inference.py` to understand how the IR is loaded and run. This repository complements this blog post: [post]()

# Instructions
1. Download and save an image of a human figure, let's call it $IMAGE_PATH
2. Follow the [Blog Post](https://medium.com/@oviyum/real-time-human-pose-estimation-on-the-edge-with-movidius-ncs-and-openvino-ac3b13536) to generate Intermediate Representations.
3. Save/Move Intermediate Representations in the root of the repository.
4. Run `python3 run_inference.py -m ./model-mobilenet_v1_075.xml -d MYRIAD -i $IMAGE_PATH`

# Additional Information
- You can run `050`, and `100` versions of the model by replacing `-i` argument of step 4 under "Instructions".

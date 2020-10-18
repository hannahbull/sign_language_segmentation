# Segment Sign Language Video

This code enables segmentation of sign language video into subtitle-units, i.e. segments of video approximately corresponding to a sentence or phrase in a subtitle. Details of the model are provided in: 

# Input

The input is in the form of series of OpenPose 2D keypoints. To clean the OpenPose 2D keypoints, use the code provided at: 

[https://github.com/hannahbull/clean_op_data_sl](https://github.com/hannahbull/clean_op_data_sl)

# Output

Subtitle file (.srt) with time tags corresponding to each detected sign language segment. 

# Example 

```python apply_model.py --which_keypoints 'body'```

# References

OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

mmskeleton: [https://github.com/open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton)

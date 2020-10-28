# Segment Sign Language Video

This code enables segmentation of sign language video into subtitle-units, i.e. segments of video approximately corresponding to a sentence or phrase in a subtitle. Details of the model can be found here: [https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf](https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf)

If this code is of use to you, please cite the following article: 

> Bull, H., Gouiffès, M., Braffort, A.: Automatic Segmentation of Sign Language into Subtitle-Units. In: Proceedings of the European Conference on Computer Vision (ECCV), Sign Language Recognition, Translation and Production (SLRTP) Workshop (2020). 

```tex
@article{bull2020automatic,
    author = {Bull, Hannah and Gouiffès, Michèle and Braffort, Annelies},
    journal = {Proceedings of the European Conference on Computer Vision (ECCV), Sign Language Recognition, Translation and Production (SLRTP) Workshop},
    month = {8},
    title = {{Automatic Segmentation of Sign Language into Subtitle-Units}},
    url = {https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf},
    year = {2020}
}
```

# Input

The input is in the form of sequences of OpenPose 2D keypoints. To clean the OpenPose 2D keypoints and to extract the sequences of likely signers, use the code provided at: 

[https://github.com/hannahbull/clean_op_data_sl](https://github.com/hannahbull/clean_op_data_sl)

# Output

Frame-level probablities of Subtitle-Units. 

Subtitle file (.srt) with time tags corresponding to each detected sign language segment. 

# Example 

An example of applying this model to extract the Subtitle-Units for a YouTube video is provided at [this Google Colab link](https://colab.research.google.com/drive/1YAfwTycO2ZvDGFHwbx5pSmHbpAlcOylN?usp=sharing)

```python apply_model.py --which_keypoints 'body'```

# References


OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

mmskeleton: [https://github.com/open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton)

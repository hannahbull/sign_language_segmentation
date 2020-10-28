# Segment Sign Language Video

This code enables segmentation of sign language video into subtitle-units, i.e. segments of video approximately corresponding to a sentence or phrase in a subtitle. Details of the model can be found here: [https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf](https://slrtp.com/papers/full_papers/SLRTP.FP.01.011.paper.pdf)

If this code is of use to you, please cite the following article: 

> Bull, H., Gouiffès, M., Braffort, A.: Automatic Segmentation of Sign Language into Subtitle-Units. In: Proceedings of the European Conference on Computer Vision (ECCV), Sign Language Recognition, Translation and Production (SLRTP) Workshop (2020)

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

# Data used to train the model 

The data used to train the model is MEDIAPI-SKEL, a 2D-skeleton database of French Sign Language video with aligned French subtitles, available on [Ortolang](https://www.ortolang.fr/market/corpora/mediapi-skel/) for research purposes. 

> Bull, H., Braffort, A., Gouiffès, M.: MEDIAPI-SKEL - a 2D-skeleton video database of French Sign Language with aligned French subtitles. In: Proceedings of the Twelfth International Conference on Language Resources and Evaluation(LREC’20). pp. 6063–6068. European Language Resource Association (ELRA), Marseille, France (May 2020)

```tex
@inproceedings{bull2020mediapiskel,
  title={{MEDIAPI}-{SKEL} - A 2{D}-Skeleton Video Database of French Sign Language With Aligned French Subtitles},
  author={Bull, Hannah and Braffort, Annelies and Gouiff\`es, Mich\`ele},
  booktitle={Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC'20)},
  year={2020},
  address = {Marseille, France},
  month = {May},
  pages ={6063--6068},
  publisher = {European Language Resource Association (ELRA)},
}
```

# Input

The input is in the form of sequences of OpenPose 2D keypoints. To clean the OpenPose 2D keypoints and to extract the sequences of likely signers, use the code provided at: 

[https://github.com/hannahbull/clean_op_data_sl](https://github.com/hannahbull/clean_op_data_sl)

# Output

Frame-level probablities of Subtitle-Units. 

Subtitle file (.srt) with time tags corresponding to each detected sign language segment. 

# Example 

An example of applying this model to extract the Subtitle-Units for a YouTube video is provided at [this Google Colab link](https://colab.research.google.com/drive/1YAfwTycO2ZvDGFHwbx5pSmHbpAlcOylN?usp=sharing). 

To produce .srt files with Subtitle-Units for sequences of OpenPose 2D keypoints, run: 

```python apply_model.py --input_folder 'data' --output_folder 'output' --which_keypoints 'body' --fps 25```

To obtain predictions for the MEDIAPI-SKEL test set, run [clean_op_data_sl](https://github.com/hannahbull/clean_op_data_sl) on the OpenPose 2D keypoints and place the folders containing the `.pkl` files in `mediapiskel_data/skeleton_sequences`. Place the `.vtt` subtitles in the folder `mediapiskel_data/subtitles`. Run: 

```python reproduce_results_mediapiskel.py --input_folder 'mediapiskel_data/skeleton_sequences' --which_keypoints 'full' --video_information 'mediapiskel_data/video_information.csv' --subtitle_folder 'mediapiskel_data/subtitles'```

# References

OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

mmskeleton: [https://github.com/open-mmlab/mmskeleton](https://github.com/open-mmlab/mmskeleton)

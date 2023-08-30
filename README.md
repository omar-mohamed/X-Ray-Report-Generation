# X-Ray-Report-Generation (VSGRU)
This is the implementation of the 'VSGRU' model mentioned in our paper 'Automated Radiology Report Generation using Conditioned Transformers'.

Paper link [here](https://doi.org/10.1016/j.imu.2021.100557).

We automatically generate full radiology reports given chest X-ray images from the IU-X-Ray dataset by conditioning a recurrent neural net on the visual and semantic features of the image.

![vsgru dpi](https://user-images.githubusercontent.com/6074821/113486170-a8614000-94b1-11eb-9050-4ebba0f94e07.png)

## Installation & Usage
*The project was tested on a virtual environment of python 3.7, pip 23.2.1, and MacOS*
- pip install -r full_requirements.txt (or pip install -r requirements.txt if there are errors because of using a different operating system, as requirements.txt only contains the main dependencies and pip will fetch the compatible sub-dependencies, but it will be slower)
- nlg-eval --setup
- python get_iu_xray.py (to download the dataset)
- python train.py

## Related Repositories
- CDGPT2 repository (main paper repo) [here](https://github.com/omar-mohamed/GPT2-Chest-X-Ray-Report-Generation).
- Finetuned Chexnet repository [here](https://github.com/omar-mohamed/Chest-X-Ray-Tags-Classification).


## Citation
To cite this paper, please use:

```
@article{ALFARGHALY2021100557,
title = {Automated radiology report generation using conditioned transformers},
journal = {Informatics in Medicine Unlocked},
volume = {24},
pages = {100557},
year = {2021},
issn = {2352-9148},
doi = {https://doi.org/10.1016/j.imu.2021.100557},
url = {https://www.sciencedirect.com/science/article/pii/S2352914821000472},
author = {Omar Alfarghaly and Rana Khaled and Abeer Elkorany and Maha Helal and Aly Fahmy}
}
```

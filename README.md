# CTSAN
This is the implementation of Co-Training Subdomain Adverserial Network for semi-supervised EEG-based Emotion Recognition in PyTorch (Version 2.6.0).

This repository contains the source code of our paper, using the following datasets:

- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html): 15 subjects participated in experiments with videos as emotion stimuli (three emotions: positive/negative/neutral) and EEG was recorded with 62 channels at a sampling rate of 1000Hz.

- [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html): 15 subjects participated experiments with videos as emotion stimuli (four emotions: happy/sad/neutral/fear).  62 EEG recordings were collected with a sampling frequency of 1000Hz.

# Install

```
pip install -r ./requirements.txt
```

# Usage

```
python ctsan.py
```

# Acknowledgement
The implementation code of domain adversarial training is bulit on the [tllib](https://github.com/thuml/Transfer-Learning-Library) code base

# Contact
If you have any questions, please contact me at [2350353412wqts@gmail.com](2350353412wqts@gmail.com).
# NYCU Data Mining(Spring 2026) Assignment 3
- Student ID: 314551087
- Name: 黃奕睿

### Environment Setup

```bash
pip install -r requirements.txt
```

### File Structure

```text
.
├── fulltrain_multiscale_se_resnet1d.py   
├── data/
│	├── train/                       
│	├── test/                                             
│	└── sample_submission.csv           
└── Others/                             
	├── DM3_Inception.ipynb                     
	├── DM3_LSTM.ipynb                     
	└── DM3_TCN.ipynb                                       
```

### Usage

```bash
python fulltrain_multiscale_se_resnet1d.py # main approach
```

## Performance

| Model | Scores | 
|------|------|
| LSTM | 0.6875 |
| TCN | 0.7318|
| Inception | 0.7598 |
| ResNet1D | 0.7824 |
| Hybrid Multi-Scale SE-ResNet1D | 0.7940 |
| Hybrid Multi-Scale SE-ResNet1D (w/o Val) | 0.8077 |

## Performance snapshot
<img src="Others/score.png" width="1000">

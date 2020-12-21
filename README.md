# NLP_Project

#### Quickstart

```python3
git clone https://github.com/cwhao98/nlp_project.git
cd nlp_project
conda create -n yourname python=3.8
source activate yourname
pip install torch
python train.py
```

#### Prerequisites

```shell
Python 3.8
Pytorch >= 1.7.0
```

#### Code

###### 二分类

```python3
python train.py —num_emotion 2
```

###### 多分类

```python3
python train.py —num_emotion 8
```


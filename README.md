# signal_recover-diffusion
A project for recovering signals from noise.

![generate_signal](assets/generate_signal.png)

## Installation

To install the project, you need:

1. `git clone https://github.com/DXWK73/signal_recover-diffusion.git`
2. `pip install -r requirements.txt`

## Train

To train yourself datas, you need:

1.Put your data into `/data/1train_data`

2.Run trian.py
```commandline
python train.py --epochs 512 --lr 1e-4 --batch_size 4 --device cuda:0
```  

3.When the train process finish, your model weight will be saved at `ckpt\sgr\model.pth`

4.Trian finish.

## Inference

To inference the model for generating signals, you need:

1.Run infer.py
```commandline
python infer.py --diffusion_steps 50 --device cuda:0 
```

2.Then an img and .txt file will be saved at `assets`

3.Inference finish.



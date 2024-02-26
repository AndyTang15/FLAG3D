# AGCN, PoseC3D  MindSpore


## Project Description:

Based on [FLAG3D](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_FLAG3D_A_3D_Fitness_Activity_Dataset_With_Language_Instruction_CVPR_2023_paper.pdf) dataset，using Mindspore framework to implement  [2s-AGCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf) and [PoseC3D](https://openaccess.thecvf.com/content/CVPR2022/papers/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf) two models for video human motion prediction.

## Environmental Description:

+ mindspore version: 2.1.1
+ Hardware platform: GPU CUDA 11.6
+ Operating system: Linux-x86_64
+ Programming language: Python 3.8

​		Environment installation command:

```
bash environment.sh
```

## File Description:

+ chpk_resume：Model weights
+ dataset：Dataset implementation
+ evaluation：Evaluation function
+ logs：Log function
+ model：Model implementation
+ test：Testing files
+ environment.sh：Environment installation command
+ train：Training files


## Directory format:

```
--data
	--FLAG
		--flag2d.pkl
		--flag3d.pkl
--gcn-c3d-mindspore
	--chpk_resume
		--agcn_2d.ckpt
		--agcn_3d.ckpt
		--posec3d_2d.ckpt
	--dataset
	--evaluation
	--logs
	--model
	--test
	--environment.sh
	--train_2sagcn_flag2d.py
	--train_2sagcn_flag3d.py
	--train_posec3d_flag2d.py
```

## Test command:

Run the following command in the gcn-c3d-mindspore directory:

```python
python ./test/test_2sagcn_2d.py # Testing the accuracy of agcn on FLAG3D(Out-domain)
python ./test/test_2sagcn_3d.py # Testing the accuracy of agcn on FLAG3D(In-domain)
python ./test/test_PoseC3D_2d.py # Testing the accuracy of PoseC3D on FLAG3D(Out-domain)
```

## Train command:

Run the following command in the gcn-c3d-mindspore directory:

```python
python train_2sagcn_flag2d.py # Training agcn on FLAG3D(Out-domain)
python train_2sagcn_flag3d.py # Training agcn on FLAG3D(In-domain)
python train_posec3d_flag2d.py # Training PoseC3D on FLAG3D(Out-domain)
```












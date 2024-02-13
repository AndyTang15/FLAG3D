python -m src.train.train_cvae --modelname cvae_transformer_rc_rcxyz_kl --pose_rep rot6d --lambda_kl 1e-5 --jointstype vertices --batch_size 20 --num_frames 60 --num_layers 8 --lr 0.0001 --glob --translation --no-vertstrans --dataset humanact12 --num_epochs 5000 --snapshot 100 --folder exps/humanact12
# 默认使用 cpu, 若要使用 gpu, 则可以在 src/train/train_cvae.py 中将 62 行注释
python -m src.evaluate.evaluate_cvae exps/humanact12/checkpoint_1000.ckpt --batch_size 64 --niter 20
# xxxx 表示你需要验证的在第 xxxx epoch 训练的模型

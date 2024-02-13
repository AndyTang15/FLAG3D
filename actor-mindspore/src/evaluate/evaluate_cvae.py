from src.parser.evaluation import parser
import mindspore as ms

# python -m src.evaluate.evaluate_cvae PATH/TO/checkpoint_XXXX.pth.tar --batch_size 64 --niter 20
def main():
    parameters, folder, checkpointname, epoch, niter = parser()
    # print("parameters: ", parameters, "\n folder: ", folder, "\n epoch: ", epoch, "\n niter: ", niter)
    # parameters: {'activation': 'gelu',
    #              'archiname': 'transformer',
    #              'batch_size': 64, 'dataset': 'humanact12',
    #              'debug': False, 'expname': 'exps', 'folder': 'exps/humanact12',
    #              'glob': True, 'glob_rot': [3.141592653589793, 0, 0], 'jointstype': 'vertices',
    #              'lambda_kl': 1e-05, 'lambda_rc': 1.0, 'lambda_rcxyz': 1.0, 'lambdas': {'kl': 1e-05, 'rc': 1.0, 'rcxyz': 1.0},
    #              'latent_dim': 256, 'losses': ['rc', 'rcxyz', 'kl'], 'lr': 0.0001, 'max_len': -1, 'min_len': -1,
    #              'modelname': 'cvae_transformer_rc_rcxyz_kl', 'modeltype': 'cvae',
    #              'num_epochs': 1, 'num_frames': 60, 'num_layers': 8, 'num_seq_max': 3000,
    #              'pose_rep': 'rot6d', 'sampling': 'conseq', 'sampling_step': 1, 'snapshot': 100,
    #              'translation': True, 'vertstrans': False, 'checkpointname': 'exps/humanact12/checkpoint_0001.pth.tar', 'niter': 20}
    #  folder: exps/humanact12
    #  epoch: 1
    #  niter: 20
    dataset = parameters["dataset"]
    # print(dataset)
    if dataset in ["ntu13", "humanact12"]:  # 进入
        from .gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["uestc"]:
        from .stgcn_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    ms.context.set_context(device_target="CPU")
    main()

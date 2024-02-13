import argparse
import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

from map import load_map


def pytorch2mindspore(args):
    #载入pth
    par_dict = torch.load(args.pth_path, map_location=torch.device('cpu'))
    map = load_map(args.pth_name)
    new_params_list = []

    for para in par_dict['state_dict']:
        if para in map:
            ms_name = map[para]
            parameter = par_dict['state_dict'][para]

            param_dict = {}
            param_dict['name'] = ms_name
            param_dict['data'] = Tensor(parameter.numpy())

            if ms_name.endswith('gcn.conv_ta.weight') or ms_name.endswith('gcn.conv_sa.weight'):
                n, c, v = parameter.size()
                param_dict['data'] = Tensor(parameter.view(n,c,1,v).numpy())

            new_params_list.append(param_dict)

    save_checkpoint(new_params_list, args.save_path)

def main(args):
    assert args.pth_name in ['stgcn_2d','agcn_2d','posec3d_2d','stgcn_3d','agcn_3d']

    pytorch2mindspore(args)

if __name__=="__main__":
    # FLAG2D
    # D:\\data\\work_dirs\\2d_baseline\\stgcn-lr0.1\\best_top1_acc_epoch_29.pth
    # D:\\data\\work_dirs\\2d_baseline\\2sagcn-j-lr0.1\\best_top1_acc_epoch_23.pth
    # D:\\data\\work_dirs\\2d_baseline\\posec3d-j-lr0.1\\best_top1_acc_epoch_30.pth

    # FLAG3D
    # D:\\data\\work_dirs\\3d_baseline\\stgcn-lr0.1\\best_top1_acc_epoch_22.pth
    # D:\\data\\work_dirs\\3d_baseline\\2sagcn-j-lr0.1\\best_top1_acc_epoch_27.pth

    parser = argparse.ArgumentParser(description='pth to ckpt')
    parser.add_argument('--pth_path', default="D:\\data\\work_dirs\\3d_baseline\\2sagcn-j-lr0.1\\best_top1_acc_epoch_27.pth",
                        type=str, help='where pth locate')
    parser.add_argument('--pth_name', default="agcn_3d",
                        choices = ['stgcn_2d','agcn_2d','posec3d_2d','stgcn_3d','agcn_3d'],type=str, help='which pth to transform')
    parser.add_argument('--save_path', default="./mindspore.ckpt",
                        type=str, help='where ckpt locate')
    args = parser.parse_args()
    main(args)


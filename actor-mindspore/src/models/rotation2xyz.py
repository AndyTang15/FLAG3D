# _*_ coding:utf-8 _*_
"""
@Software: ACTOR_mindspore
@FileName: rotation2xyz.py
@Date: 2023/4/27 10:27
@Author: caijianfeng
"""
import mindspore as ms
import numpy as np
import src.utils.rotation_conversions as geometry
import torch
from .smpl import SMPL, JOINTSTYPE_ROOT
from .get_model import JOINTSTYPES


class Rotation2xyz:
    def __init__(self, device=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.smpl_model = SMPL().eval()

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):
        # x = batch["x"] (shape = [batch size, 25, 6, 60]), mask = batch["mask"] (shape = [batch size, 60])
        # {"pose_rep": self.pose_rep = 'rot6d',
        #  "glob_rot": self.glob_rot = [3.141592653589793, 0, 0],
        #  "glob": self.glob = True,
        #  "jointstype": self.jointstype = 'vertices',
        #  "translation": self.translation = True,
        #  "vertstrans": self.vertstrans = False}
        if pose_rep == "xyz":  # 不会进入
            return x

        if mask is None:  # 不会进入
            # mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)
            mask = ms.ops.ones((x.shape[0], x.shape[-1]), type=ms.bool_)

        if not glob and glob_rot is None:  # 不会进入
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:  # 不会进入
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:  # 进入
            x_translations = x[:, -1, :3]  # -1 表示该维度舍弃(取该维度的最后一个)  shape = (x.shape[0], 3, x.shape[3]) = [batch size, 3, 60]
            x_rotations = x[:, :-1]  # shape = (x.shape[0], x.shape[1]-1, x.shape[2], x.shape[3]) = [batch size, 24, 6, 60]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)  # shape = [batch size, 60, 24, 6]
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":  # 进入
            # x_rotations_numpy = x_rotations.numpy()  # shape = [batch size, 60, 24, 6]
            # mask_numpy = mask.numpy()  # shape = [batch size, 60]
            # x_rotations_mask = x_rotations_numpy[mask_numpy]
            # x_rotations_mask = ms.Tensor.from_numpy(x_rotations_mask).astype(ms.float32)
            # rotations = geometry.rotation_6d_to_matrix(x_rotations_mask)  # shape = [num, 24, 3, 3]
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            # global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = ms.Tensor(glob_rot)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:  # 进入
            global_orient = rotations[:, 0]  # shape = [num, 3, 3]
            rotations = rotations[:, 1:]  # shape = [num, 23, 3, 3]

        if betas is None:  # 进入
            # betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
            #                     dtype=rotations.dtype, device=rotations.device)
            betas = ms.ops.zeros((rotations.shape[0], self.smpl_model.num_betas),
                                 rotations.dtype)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
        rotations = torch.from_numpy(rotations.numpy())
        global_orient = torch.from_numpy(global_orient.numpy())
        betas = torch.from_numpy(betas.numpy())
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        # print(out)
        # get the desirable joints
        joints = out[jointstype]

        # x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        # x_xyz = ms.numpy.empty((nsamples, time, joints.shape[1], 3), dtype=x.dtype)
        # x_xyz[~mask] = 0
        # x_xyz[mask] = joints
        x_xyz = np.zeros((nsamples, time, joints.shape[1], 3))  # shape = [batch size, 60, ?, 3]
        mask_numpy = mask.numpy()  # shape = [batch size, 60]
        x_xyz[~mask_numpy] = 0
        x_xyz[mask_numpy] = joints.numpy()
        x_xyz = ms.Tensor.from_numpy(x_xyz).astype(ms.float32)
        x_xyz = x_xyz.permute(0, 2, 3, 1)  # shape = [batch size, ?, 3, 60]

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":  # 不会进入
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:  # 不会进入 -> vertstrans = False
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        return x_xyz  # shape = [batch size, ?, 3, 60]

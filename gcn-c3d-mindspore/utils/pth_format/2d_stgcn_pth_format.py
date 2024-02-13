"""


{
    'meta': {'env_info': 'sys.platform: linux\nPython: 3.9.13 (main, Aug 25 2022, 23:26:10) [GCC 11.2.0]\nCUDA available: True\nGPU 0,1,2,3: NVIDIA A100-SXM4-80GB\n'
                         'CUDA_HOME: /mnt/petrelfs/share/cuda-11.3\nNVCC: Cuda compilation tools, release 11.3, V11.3.109\nGCC: gcc (GCC) 5.4.0\nPyTorch: 1.11.0\n'
                         'PyTorch compiling details: PyTorch built with:\n  - GCC 7.3\n  - C++ Version: 201402\n  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications\n'
                         '  - Intel(R) MKL-DNN v2.5.2 (Git Hash a9302535553c73243c632ad3c4c80beec3d19a1e)\n  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided by MKL)\n'
                         '  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime 11.3\n  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;'
                         '-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;'
                         '-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37\n'
                         '  - CuDNN 8.2\n  - Magma 2.5.2\n  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, '
                         'CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG '
                         '-DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC '
                         '-Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare '
                         '-Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations '
                         '-Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new '
                         '-Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, '
                         'PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, '
                         'USE_MKL=ON, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, \n\nTorchVision: 0.12.0\nOpenCV: 4.6.0\nMMCV: 1.6.2\nMMCV '
                         'Compiler: n/a\nMMCV CUDA Compiler: n/a\npyskl: 0.1.0+ff98fea',
            'seed': 1527494470,
            'config_name': 'stgcn.py',
            'work_dir': 'stgcn-lr0.1',
            'hook_msgs': {'last_ckpt': '/mnt/petrelfs/daiwenxun/pyskl/pyskl/work_dirs/2d_baseline/stgcn-lr0.1/epoch_29.pth', 'best_score': 0.6976388888888889,
                            'best_ckpt': '/mnt/petrelfs/daiwenxun/pyskl/pyskl/work_dirs/2d_baseline/stgcn-lr0.1/best_top1_acc_epoch_29.pth'},
            'epoch': 29,
            'iter': 9802,
            'mmcv_version': '1.6.2',
            'time': 'Tue Nov  1 17:12:53 2022'},

    'state_dict': {
backbone.data_bn.weight 	 torch.Size([51])
backbone.data_bn.bias 	 torch.Size([51])
backbone.data_bn.running_mean 	 torch.Size([51])
backbone.data_bn.running_var 	 torch.Size([51])
backbone.data_bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.0.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.0.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.0.gcn.bn.weight 	 torch.Size([64])
backbone.gcn.0.gcn.bn.bias 	 torch.Size([64])
backbone.gcn.0.gcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.0.gcn.bn.running_var 	 torch.Size([64])
backbone.gcn.0.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.0.gcn.conv.weight 	 torch.Size([192, 3, 1, 1])
backbone.gcn.0.gcn.conv.bias 	 torch.Size([192])
backbone.gcn.0.tcn.conv.weight 	 torch.Size([64, 64, 9, 1])
backbone.gcn.0.tcn.conv.bias 	 torch.Size([64])
backbone.gcn.0.tcn.bn.weight 	 torch.Size([64])
backbone.gcn.0.tcn.bn.bias 	 torch.Size([64])
backbone.gcn.0.tcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.0.tcn.bn.running_var 	 torch.Size([64])
backbone.gcn.0.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.1.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.1.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.1.gcn.bn.weight 	 torch.Size([64])
backbone.gcn.1.gcn.bn.bias 	 torch.Size([64])
backbone.gcn.1.gcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.1.gcn.bn.running_var 	 torch.Size([64])
backbone.gcn.1.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.1.gcn.conv.weight 	 torch.Size([192, 64, 1, 1])
backbone.gcn.1.gcn.conv.bias 	 torch.Size([192])
backbone.gcn.1.tcn.conv.weight 	 torch.Size([64, 64, 9, 1])
backbone.gcn.1.tcn.conv.bias 	 torch.Size([64])
backbone.gcn.1.tcn.bn.weight 	 torch.Size([64])
backbone.gcn.1.tcn.bn.bias 	 torch.Size([64])
backbone.gcn.1.tcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.1.tcn.bn.running_var 	 torch.Size([64])
backbone.gcn.1.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.2.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.2.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.2.gcn.bn.weight 	 torch.Size([64])
backbone.gcn.2.gcn.bn.bias 	 torch.Size([64])
backbone.gcn.2.gcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.2.gcn.bn.running_var 	 torch.Size([64])
backbone.gcn.2.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.2.gcn.conv.weight 	 torch.Size([192, 64, 1, 1])
backbone.gcn.2.gcn.conv.bias 	 torch.Size([192])
backbone.gcn.2.tcn.conv.weight 	 torch.Size([64, 64, 9, 1])
backbone.gcn.2.tcn.conv.bias 	 torch.Size([64])
backbone.gcn.2.tcn.bn.weight 	 torch.Size([64])
backbone.gcn.2.tcn.bn.bias 	 torch.Size([64])
backbone.gcn.2.tcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.2.tcn.bn.running_var 	 torch.Size([64])
backbone.gcn.2.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.3.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.3.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.3.gcn.bn.weight 	 torch.Size([64])
backbone.gcn.3.gcn.bn.bias 	 torch.Size([64])
backbone.gcn.3.gcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.3.gcn.bn.running_var 	 torch.Size([64])
backbone.gcn.3.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.3.gcn.conv.weight 	 torch.Size([192, 64, 1, 1])
backbone.gcn.3.gcn.conv.bias 	 torch.Size([192])
backbone.gcn.3.tcn.conv.weight 	 torch.Size([64, 64, 9, 1])
backbone.gcn.3.tcn.conv.bias 	 torch.Size([64])
backbone.gcn.3.tcn.bn.weight 	 torch.Size([64])
backbone.gcn.3.tcn.bn.bias 	 torch.Size([64])
backbone.gcn.3.tcn.bn.running_mean 	 torch.Size([64])
backbone.gcn.3.tcn.bn.running_var 	 torch.Size([64])
backbone.gcn.3.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.4.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.4.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.4.gcn.bn.weight 	 torch.Size([128])
backbone.gcn.4.gcn.bn.bias 	 torch.Size([128])
backbone.gcn.4.gcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.4.gcn.bn.running_var 	 torch.Size([128])
backbone.gcn.4.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.4.gcn.conv.weight 	 torch.Size([384, 64, 1, 1])
backbone.gcn.4.gcn.conv.bias 	 torch.Size([384])
backbone.gcn.4.tcn.conv.weight 	 torch.Size([128, 128, 9, 1])
backbone.gcn.4.tcn.conv.bias 	 torch.Size([128])
backbone.gcn.4.tcn.bn.weight 	 torch.Size([128])
backbone.gcn.4.tcn.bn.bias 	 torch.Size([128])
backbone.gcn.4.tcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.4.tcn.bn.running_var 	 torch.Size([128])
backbone.gcn.4.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.4.residual.conv.weight 	 torch.Size([128, 64, 1, 1])
backbone.gcn.4.residual.conv.bias 	 torch.Size([128])
backbone.gcn.4.residual.bn.weight 	 torch.Size([128])
backbone.gcn.4.residual.bn.bias 	 torch.Size([128])
backbone.gcn.4.residual.bn.running_mean 	 torch.Size([128])
backbone.gcn.4.residual.bn.running_var 	 torch.Size([128])
backbone.gcn.4.residual.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.5.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.5.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.5.gcn.bn.weight 	 torch.Size([128])
backbone.gcn.5.gcn.bn.bias 	 torch.Size([128])
backbone.gcn.5.gcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.5.gcn.bn.running_var 	 torch.Size([128])
backbone.gcn.5.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.5.gcn.conv.weight 	 torch.Size([384, 128, 1, 1])
backbone.gcn.5.gcn.conv.bias 	 torch.Size([384])
backbone.gcn.5.tcn.conv.weight 	 torch.Size([128, 128, 9, 1])
backbone.gcn.5.tcn.conv.bias 	 torch.Size([128])
backbone.gcn.5.tcn.bn.weight 	 torch.Size([128])
backbone.gcn.5.tcn.bn.bias 	 torch.Size([128])
backbone.gcn.5.tcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.5.tcn.bn.running_var 	 torch.Size([128])
backbone.gcn.5.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.6.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.6.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.6.gcn.bn.weight 	 torch.Size([128])
backbone.gcn.6.gcn.bn.bias 	 torch.Size([128])
backbone.gcn.6.gcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.6.gcn.bn.running_var 	 torch.Size([128])
backbone.gcn.6.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.6.gcn.conv.weight 	 torch.Size([384, 128, 1, 1])
backbone.gcn.6.gcn.conv.bias 	 torch.Size([384])
backbone.gcn.6.tcn.conv.weight 	 torch.Size([128, 128, 9, 1])
backbone.gcn.6.tcn.conv.bias 	 torch.Size([128])
backbone.gcn.6.tcn.bn.weight 	 torch.Size([128])
backbone.gcn.6.tcn.bn.bias 	 torch.Size([128])
backbone.gcn.6.tcn.bn.running_mean 	 torch.Size([128])
backbone.gcn.6.tcn.bn.running_var 	 torch.Size([128])
backbone.gcn.6.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.7.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.7.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.7.gcn.bn.weight 	 torch.Size([256])
backbone.gcn.7.gcn.bn.bias 	 torch.Size([256])
backbone.gcn.7.gcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.7.gcn.bn.running_var 	 torch.Size([256])
backbone.gcn.7.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.7.gcn.conv.weight 	 torch.Size([768, 128, 1, 1])
backbone.gcn.7.gcn.conv.bias 	 torch.Size([768])
backbone.gcn.7.tcn.conv.weight 	 torch.Size([256, 256, 9, 1])
backbone.gcn.7.tcn.conv.bias 	 torch.Size([256])
backbone.gcn.7.tcn.bn.weight 	 torch.Size([256])
backbone.gcn.7.tcn.bn.bias 	 torch.Size([256])
backbone.gcn.7.tcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.7.tcn.bn.running_var 	 torch.Size([256])
backbone.gcn.7.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.7.residual.conv.weight 	 torch.Size([256, 128, 1, 1])
backbone.gcn.7.residual.conv.bias 	 torch.Size([256])
backbone.gcn.7.residual.bn.weight 	 torch.Size([256])
backbone.gcn.7.residual.bn.bias 	 torch.Size([256])
backbone.gcn.7.residual.bn.running_mean 	 torch.Size([256])
backbone.gcn.7.residual.bn.running_var 	 torch.Size([256])
backbone.gcn.7.residual.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.8.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.8.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.8.gcn.bn.weight 	 torch.Size([256])
backbone.gcn.8.gcn.bn.bias 	 torch.Size([256])
backbone.gcn.8.gcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.8.gcn.bn.running_var 	 torch.Size([256])
backbone.gcn.8.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.8.gcn.conv.weight 	 torch.Size([768, 256, 1, 1])
backbone.gcn.8.gcn.conv.bias 	 torch.Size([768])
backbone.gcn.8.tcn.conv.weight 	 torch.Size([256, 256, 9, 1])
backbone.gcn.8.tcn.conv.bias 	 torch.Size([256])
backbone.gcn.8.tcn.bn.weight 	 torch.Size([256])
backbone.gcn.8.tcn.bn.bias 	 torch.Size([256])
backbone.gcn.8.tcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.8.tcn.bn.running_var 	 torch.Size([256])
backbone.gcn.8.tcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.9.gcn.PA 	 torch.Size([3, 17, 17])
backbone.gcn.9.gcn.A 	 torch.Size([3, 17, 17])
backbone.gcn.9.gcn.bn.weight 	 torch.Size([256])
backbone.gcn.9.gcn.bn.bias 	 torch.Size([256])
backbone.gcn.9.gcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.9.gcn.bn.running_var 	 torch.Size([256])
backbone.gcn.9.gcn.bn.num_batches_tracked 	 torch.Size([])
backbone.gcn.9.gcn.conv.weight 	 torch.Size([768, 256, 1, 1])
backbone.gcn.9.gcn.conv.bias 	 torch.Size([768])
backbone.gcn.9.tcn.conv.weight 	 torch.Size([256, 256, 9, 1])
backbone.gcn.9.tcn.conv.bias 	 torch.Size([256])
backbone.gcn.9.tcn.bn.weight 	 torch.Size([256])
backbone.gcn.9.tcn.bn.bias 	 torch.Size([256])
backbone.gcn.9.tcn.bn.running_mean 	 torch.Size([256])
backbone.gcn.9.tcn.bn.running_var 	 torch.Size([256])
backbone.gcn.9.tcn.bn.num_batches_tracked 	 torch.Size([])
cls_head.fc_cls.weight 	 torch.Size([60, 256])
cls_head.fc_cls.bias 	 torch.Size([60])

    }


    'optimizer' :{
    state :dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101])
    param_groups: [{'lr': 0.0002755268777791697, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0.0005, 'nesterov': True, 'maximize': False, 'initial_lr': 0.1, 'params': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]}]
    }


}


"""
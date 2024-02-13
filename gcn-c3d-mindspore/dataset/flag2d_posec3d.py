import pickle
import mindspore.dataset as ds

from dataset.transform_posec3d import UniformSampleFrames, PoseDecode, PoseCompact, Resize, \
    RandomResizedCrop, Flip, GeneratePoseTarget, FormatShape, Collect, ToTensor


left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

PoseDecode = PoseDecode()
PoseCompact = PoseCompact(hw_ratio=1., allow_imgpad=True)
RandomResizedCrop = RandomResizedCrop(area_range=(0.56, 1.0))
Flip = Flip(flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp)
Resize_train1 = Resize(scale=(-1, 64))
Resize_train2 = Resize(scale=(56, 56), keep_ratio=False)
Resize_val_test = Resize(scale=(64, 64), keep_ratio=False)
Resize_val_test = Resize(scale=(64, 64), keep_ratio=False)
GeneratePoseTarget_train_val = GeneratePoseTarget(with_kp=True, with_limb=False)
GeneratePoseTarget_test = GeneratePoseTarget(with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp)
FormatShape = FormatShape(input_format='NCTHW_Heatmap')
Collect = Collect(keys=['imgs', 'label'])
ToTensor_train = ToTensor(keys=['imgs', 'label'])
ToTensor_val_test = ToTensor(keys=['imgs'])

"""
pipeline:
origin              dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score'])
UniformSampleFrames dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips'])
PoseDecode          dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips'])
PoseCompact         dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips', 'crop_quadruple'])
Resize_val_test     dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips', 'crop_quadruple', 'scale_factor', 'keep_ratio'])
GeneratePoseTarget_test dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips', 'crop_quadruple', 'scale_factor', 'keep_ratio', 'imgs'])
FormatShape         dict_keys(['frame_dir', 'label', 'img_shape', 'original_shape', 'total_frames', 'keypoint', 'keypoint_score', 'frame_inds', 'clip_len', 'frame_interval', 'num_clips', 'crop_quadruple', 'scale_factor', 'keep_ratio', 'imgs', 'input_shape'])
Collect             dict_keys(['imgs', 'label'])
ToTensor_val_test   dict_keys(['imgs', 'label'])
"""

class FLAG2DPoseC3DTrainDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 1, test_mode=False):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['train'])
        self.dataset = self.dataset['annotations'][:self.dataset_len]#[:self.dataset_len]

        # origin: (1, 1045, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 1045, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = PoseCompact.transform(self.dataset[i])
            self.dataset[i] = Resize_train1.transform(self.dataset[i])
            self.dataset[i] = RandomResizedCrop.transform(self.dataset[i])
            self.dataset[i] = Resize_train2.transform(self.dataset[i])
            self.dataset[i] = Flip.transform(self.dataset[i])
            self.dataset[i] = GeneratePoseTarget_train_val.transform(self.dataset[i])
            self.dataset[i] = FormatShape.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor_train.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['imgs'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num

class FLAG2DPoseC3DValDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 1, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['val']) # 7200
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train']):]

        # origin: (1, 1045, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 1045, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = PoseCompact.transform(self.dataset[i])
            self.dataset[i] = Resize_val_test.transform(self.dataset[i])
            self.dataset[i] = GeneratePoseTarget_train_val.transform(self.dataset[i])
            self.dataset[i] = FormatShape.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor_val_test.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['imgs'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num

class FLAG2DPoseC3DValDatasetGeneratorSingleGPU():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, step, size, clip_len = 500, num_clips = 1, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['val']) # 7200
        self.size = size
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train'])+step * size:len(self.dataset['split']['train'])+(step+1) * size]

        # origin: (1, 1045, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 1045, 17, 3)

        for i in range(size):
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = PoseCompact.transform(self.dataset[i])
            self.dataset[i] = Resize_val_test.transform(self.dataset[i])
            self.dataset[i] = GeneratePoseTarget_train_val.transform(self.dataset[i])
            self.dataset[i] = FormatShape.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor_val_test.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['imgs'], self.dataset[index]['label']

    def __len__(self):
        return self.size

    def class_num(self):
        return self.class_num

class FLAG2DPoseC3DTestDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 100, num_clips = 2, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['val'])
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train']):]

        # origin: (1, 745, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode)

        for i in range(self.dataset_len):
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i]) # (1, 745, 17, 2)
            self.dataset[i] = PoseDecode.transform(self.dataset[i]) # (1, 200, 17, 2)
            self.dataset[i] = PoseCompact.transform(self.dataset[i]) # (1, 200, 17, 2)
            self.dataset[i] = Resize_val_test.transform(self.dataset[i]) # (1, 200, 17, 2)
            self.dataset[i] = GeneratePoseTarget_test.transform(self.dataset[i]) # (400, 17, 64, 64)
            self.dataset[i] = FormatShape.transform(self.dataset[i]) # (4, 17, 100, 64, 64)
            self.dataset[i] = Collect.transform(self.dataset[i]) # (4, 17, 100, 64, 64)
            self.dataset[i] = ToTensor_val_test.transform(self.dataset[i]) # (4, 17, 100, 64, 64)

    def __getitem__(self, index):
        return self.dataset[index]['imgs'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num

if __name__=="__main__":
    dataset_generator = FLAG2DPoseC3DTestDatasetGenerator("D:\\data\\flag2d.pkl")
    dataset = ds.GeneratorDataset(dataset_generator, ["imgs", "label"], shuffle=True).batch(2, True)
    for data in dataset.create_dict_iterator():
        # Train: Tensor(32, 1, 17, 500, 56, 56) Tensor(32)
        # Val: Tensor(32, 1, 17, 500, 64, 64) Tensor(32)
        # Test: Tensor(32, 20, 17, 500, 64, 64) Tensor(32) # 经过double
        # (batch_size, num_clip(double), num_keypoint, num_frames, h, w)
        print(data["imgs"].shape, data["label"])
import numpy as np
import pickle
import mindspore.dataset as ds

from dataset.transform import PreNormalize2D, GenSkeFeat, UniformSampleFrames, PoseDecode, FormatGCNInput, Collect, \
    ToTensor

PreNormalize2D = PreNormalize2D(img_shape = (480, 854)) # (1, 1045, 17, 2)
GenSkeFeat = GenSkeFeat() # (1, 1045, 17, 3)
PoseDecode = PoseDecode() # (1, 500, 17, 3)
FormatGCNInput = FormatGCNInput() # (1, 1, 500, 17, 3)
Collect = Collect() # (1, 1, 500, 17, 3)
ToTensor = ToTensor() # (1, 1, 500, 17, 3)

class FLAG2DTrainDatasetGenerator():
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
            self.dataset[i] = PreNormalize2D.transform(self.dataset[i])
            self.dataset[i] = GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num

class FLAG2DValDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 1, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['val'])
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train']):]

        # origin: (1, 745, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 745, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = PreNormalize2D.transform(self.dataset[i])
            self.dataset[i] = GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        # Tensor:  int:
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num

class FLAG2DTestDatasetGenerator():
    """
    dataset_dir: where the dataset locate
    """
    def __init__(self, dataset_dir, clip_len = 500, num_clips = 10, test_mode=True):
        with open(dataset_dir, "rb") as dataset:
            self.dataset = pickle.load(dataset) # No changed format dataset

        self.class_num = 60
        self.keypoint_num = 17
        self.dataset_len = len(self.dataset['split']['val'])
        self.dataset = self.dataset['annotations'][len(self.dataset['split']['train']):]

        # origin: (1, 745, 17, 2)
        self.UniformSampleFrames = UniformSampleFrames(clip_len, num_clips, test_mode) # (1, 745, 17, 3)

        for i in range(self.dataset_len):
            self.dataset[i] = PreNormalize2D.transform(self.dataset[i])
            self.dataset[i] = GenSkeFeat.transform(self.dataset[i])
            self.dataset[i] = self.UniformSampleFrames.transform(self.dataset[i])
            self.dataset[i] = PoseDecode.transform(self.dataset[i])
            self.dataset[i] = FormatGCNInput.transform(self.dataset[i])
            self.dataset[i] = Collect.transform(self.dataset[i])
            self.dataset[i] = ToTensor.transform(self.dataset[i])

    def __getitem__(self, index):
        # Tensor:  int:
        return self.dataset[index]['keypoint'], self.dataset[index]['label']

    def __len__(self):
        return self.dataset_len

    def class_num(self):
        return self.class_num


if __name__=="__main__":
    dataset_generator = FLAG2DTestDatasetGenerator("D:\\data\\flag2d.pkl")
    dataset = ds.GeneratorDataset(dataset_generator, ["keypoint", "label"], shuffle=True).batch(2, True)
    for data in enumerate(dataset):
    # for data in dataset.create_dict_iterator():
        # Tensor(32, 10, 1, 500, 17, 3) Tensor(32)
        # (Batch_size, num_clips, num_person, frames, num_keypoint, keypoint_location+keypoint_score) (label)
        print(data)
        # print(data["keypoint"].shape, data["label"])
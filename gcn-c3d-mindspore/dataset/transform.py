import numpy as np
from mindspore.common.tensor import Tensor


class PreNormalize2D():
    """Normalize the range of keypoint values.

    Required Keys:

        - keypoint
        - img_shape (optional)

    Modified Keys:

        - keypoint

    Args:
        img_shape (tuple[int, int]): The resolution of the original video.
            Defaults to ``(1080, 1920)``.
    """

    def __init__(self, img_shape: tuple = (1080, 1920)) -> None:
        self.img_shape = img_shape

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PreNormalize2D`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        h, w = results.get('img_shape', self.img_shape)
        results['keypoint'][..., 0] = \
            (results['keypoint'][..., 0] - (w / 2)) / (w / 2)
        results['keypoint'][..., 1] = \
            (results['keypoint'][..., 1] - (h / 2)) / (h / 2)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'img_shape={self.img_shape})')
        return repr_str

class GenSkeFeat():
    """Unified interface for generating multi-stream skeleton features."""
    def transform(self, results: dict) -> dict:
        if 'keypoint_score' in results and 'keypoint' in results:
            results['keypoint'] = np.concatenate(
                    [results['keypoint'], results['keypoint_score'][..., None]], -1)
        return results

class UniformSampleFrames():
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False,
                 seed: int = 255) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_test_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for testing clips.
        """

        np.random.seed(self.seed)
        all_inds = []
        for i in range(self.num_clips):
            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips \
                    else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`UniformSampleFrames`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int64)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(np.int32) # 添加索引到字典中
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

class PoseDecode():
    """Load and decode pose with given indices."""
    def _load_kp(self, kp: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoints according to sampled indexes."""
        return kp[:, frame_inds].astype(np.float32)


    def _load_kpscore(self, kpscore: np.ndarray,
                      frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoint scores according to sampled indexes."""
        return kpscore[:, frame_inds].astype(np.float32)

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`PoseDecode`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'total_frames' not in results:
            results['total_frames'] = results['keypoint'].shape[1]

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(
                results['keypoint_score'], frame_inds)

        results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

class FormatGCNInput():
    """Format final skeleton shape.
    """

    def __init__(self, num_person: int = 1, mode: str = 'zero') -> None:
        self.num_person = num_person # 人数
        assert mode in ['zero', 'loop']
        self.mode = mode # zero padding

    def transform(self, results: dict) -> dict:
        """The transform function of :class:`FormatGCNInput`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results['keypoint']

        # # 之前已经concat过一次'keypoint_score'了
        #
        # if 'keypoint_score' in results:
        #     keypoint = np.concatenate(
        #         (keypoint, results['keypoint_score'][..., None]), axis=-1)

        cur_num_person = keypoint.shape[0]

        # FLAG3D是一个人的，所以这里没用上
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros(
                (pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and cur_num_person == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif cur_num_person > self.num_person:
            keypoint = keypoint[:self.num_person]

        # 根据num_clips改变下形状
        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape(
            (M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

class Collect():
    """Collect keypoint and label"""
    def __init__(self):
        self.keys = ['keypoint', 'label']

    def transform(self, results: dict) -> dict:
        results_back = {}
        for key in self.keys:
            results_back[key] = results[key]

        return results_back

class ToTensor():
    """ToTensor"""
    def __init__(self):
        self.keys = ['keypoint']

    def transform(self, results: dict) -> dict:
        results[self.keys[0]]=Tensor(results['keypoint'])
        return results


if __name__=="__main__":
    GenSkeFeat = GenSkeFeat(dataset='coco', feats=['j'])

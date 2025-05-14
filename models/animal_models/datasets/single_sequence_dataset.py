import os
from pathlib import Path
from glob import glob
import random
import re
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode

import sys
submodule_dir = os.path.abspath("externals/3DAnimals")
sys.path.append(submodule_dir)

from externals.Animals.model.dataset.util import *


class BaseSequenceDataset(Dataset):
    def __init__(self, root, skip_beginning=4, skip_end=4, local_dir=None):
        super().__init__()
        self.skip_beginning = skip_beginning
        self.skip_end = skip_end
        if local_dir is not None:
            root = copy_data_to_local(root, local_dir)
        self.sequences = self._make_single_sequence(root)
        self.samples = []

    def _make_single_sequence(self, path):
        result = []
        files = self._parse_folder(path)
        result.append(files)
        return result

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '*' + image_path_suffix)))
        img_ext = [os.path.splitext(image_path_suffix)[-1]]
        if ".*" in img_ext:
            img_ext = set([os.path.splitext(r)[-1] for r in result])
        all_result = []
        for ext in img_ext:
            all_result += [p.replace(image_path_suffix.replace(".*", ext), '{}') for p in result if ext in p]
        all_result = sorted(list(set(all_result)))
        if len(all_result) <= self.skip_beginning + self.skip_end:
            return []
        if self.skip_end == 0:
            return all_result[self.skip_beginning:]
        return all_result[self.skip_beginning:-self.skip_end]

    def _load_ids(self, path_patterns, loader, transform=None):
        result = []
        for p in path_patterns:
            suffix = loader[0]
            if ".*" in suffix:
                re_pattern = re.compile(p.format(suffix))
                all_occurrence = [str(f) for f in Path(p).parent.iterdir() if re.search(re_pattern, str(f))]
                suffix = suffix.replace(".*", os.path.splitext(all_occurrence[0])[-1])
            x = loader[1](p.format(suffix), *loader[2:])
            if transform:
                x = transform(x)
            result.append(x)
        return tuple(result)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplemented("This is a base class and should not be used directly")


class SingleSequenceDataset(BaseSequenceDataset):
    def __init__(self, root, num_frames=2, skip_beginning=False, skip_end=False, in_image_size=256, load_keypoint=False,
                 out_image_size=256, random_sample=False, dense_sample=True, load_flow=False, load_depth=False,
                 load_articulation=False, load_dino_feature=True, dino_feature_dim=16, local_dir=None,
                 ):
        self.image_loader = ["rgb.png", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.metadata_loader = ["metadata.json", metadata_loader]
        super().__init__(root, skip_beginning, skip_end, local_dir=local_dir)
        if load_flow and num_frames > 1:
            self.flow_loader = ["flow.png", cv2.imread, cv2.IMREAD_UNCHANGED]
        else:
            self.flow_loader = None

        self.num_frames = num_frames
        self.random_sample = random_sample
        if self.random_sample:
            self.samples = self.sequences
        else:
            for i, s in enumerate(self.sequences):
                stride = 1 if dense_sample else self.num_frames
                self.samples += [(i, k) for k in range(0, len(s), stride)]
            if dense_sample:  # duplicate head and tail if dense samples
                self.samples = [self.samples[0]] * num_frames + self.samples + [self.samples[-self.num_frames]] * num_frames

        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.out_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose(
            [transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        if self.flow_loader is not None:
            def flow_transform(x):
                x = torch.FloatTensor(x.astype(np.float32)).flip(2)[:, :, :2]  # HxWx2
                x = x / 65535. * 2 - 1  # -1~1
                x = \
                torch.nn.functional.interpolate(x.permute(2, 0, 1)[None], size=self.out_image_size, mode="bilinear")[
                    0]  # 2xHxW
                return x
            self.flow_transform = flow_transform
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feature.png", dino_loader, dino_feature_dim]
        self.load_flow = load_flow
        self.load_depth = load_depth
        self.load_articulation = load_articulation
        self.load_keypoint = load_keypoint
        if self.load_keypoint:
            self.keypoint_loader = ["keypoint.txt", keypoint_loader]

    def __getitem__(self, index):
        if self.random_sample:
            seq_idx = index % len(self.samples)
            seq = self.samples[seq_idx]
            if len(seq) < self.num_frames:
                start_frame_idx = 0
            else:
                start_frame_idx = np.random.randint(len(seq) - self.num_frames + 1)
        else:
            seq_idx, start_frame_idx = self.samples[index % len(self.samples)]
            seq = self.sequences[seq_idx]
            ## handle edge case: when only last frame is left, sample last two frames, except if the sequence only has one frame
            if len(seq) <= start_frame_idx + 1:
                start_frame_idx = max(0, start_frame_idx - 1)

        paths = seq[start_frame_idx:start_frame_idx + self.num_frames]  # length can be shorter than num_frames
        images = torch.stack(self._load_ids(paths, self.image_loader, transform=self.image_transform),
                             0)  # load all images
        masks = torch.stack(self._load_ids(paths, self.mask_loader, transform=self.mask_transform),
                            0)  # load all images
        mask_dt = compute_distance_transform(masks)
        metadata_seq = self._load_ids(paths, self.metadata_loader)
        global_frame_ids = torch.LongTensor([int(metadata.get("clip_frame_id")) for metadata in metadata_seq])
        xmins, ymins, xmaxs, ymaxs = map(
            torch.Tensor, zip(*[metadata.get("crop_box_xyxy") for metadata in metadata_seq])
        )
        try:
            full_ws = torch.Tensor([metadata.get("video_frame_width") for metadata in metadata_seq])
            full_hs = torch.Tensor([metadata.get("video_frame_height") for metadata in metadata_seq])
        except Exception:
            try:
                full_ws = torch.Tensor([metadata.get("frame_size_wh")[0] for metadata in metadata_seq])
                full_hs = torch.Tensor([metadata.get("frame_size_wh")[1] for metadata in metadata_seq])
            except Exception:
                full_ws = torch.Tensor([1920] * len(paths))
                full_hs = torch.Tensor([1080] * len(paths))
        bboxs = torch.stack(
            [global_frame_ids, xmins, ymins, xmaxs - xmins, ymaxs - ymins, full_ws, full_hs, torch.zeros(len(paths))]).T
        mask_valid = get_valid_mask(bboxs, (
        self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        if self.load_flow and len(paths) > 1:
            flows = torch.stack(self._load_ids(paths[:-1], self.flow_loader, transform=self.flow_transform),
                                0)  # load flow from current frame to next, (N-1)x(x,y)xHxW, -1~1
        else:
            flows = None
        bg_images = None
        if self.load_dino_feature:
            dino_features = torch.stack(self._load_ids(paths, self.dino_feature_loader, transform=torch.FloatTensor),
                                        0)  # Fx64x224x224
        else:
            dino_features = None
        dino_clusters = None
        seq_idx = torch.LongTensor([seq_idx])
        frame_idx = torch.arange(start_frame_idx, start_frame_idx + len(paths)).long()
        if self.load_depth:
            depth = self._load_ids(paths, self.depth_loader, transform=self.depth_transform).unsqueeze(0)
            depth = normalize_depth(depth, masks[:, [0]])
        else:
            depth = None
        if self.load_articulation:
            articulation, articulation_flag = self._load_ids(paths, self.articulation_loader)
            if articulation is not None:
                if self.reverse_articulation:
                    articulation_reversed = torch.zeros_like(articulation)
                    for idx1, idx2 in self.reverse_idx_pairs:
                        articulation_reversed[idx1] = -articulation[idx2]
                        articulation_reversed[idx2] = -articulation[idx1]
                    articulation_reversed[:, 2] = -articulation_reversed[:, 2]  # Do not reverse dimension 2
                    articulation = articulation_reversed
                articulation = articulation.unsqueeze(0)
        else:
            articulation, articulation_flag = None, None
        if self.load_keypoint:
            keypoint = self._load_ids(paths, self.keypoint_loader)
            keypoint = torch.stack(keypoint) / self.in_image_size * 2 - 1
        else:
            keypoint = None

        ## pad shorter sequence
        if len(paths) < self.num_frames:
            num_pad = self.num_frames - len(paths)
            pad_front = lambda x: None if x is None else torch.cat([x[:1]] * num_pad + [x], 0)
            images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint, frame_idx= (
            *map(pad_front, (
            images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint, frame_idx)),)
            if flows is not None:
                flows[:num_pad] = 0  # setting flow to zeros for replicated frames
            paths = [paths[0]] * num_pad + paths

        out = (*map(none_to_nan, (
        images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint, seq_idx, frame_idx, paths)),)  # for batch collation
        return out

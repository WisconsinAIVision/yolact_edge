import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils
import contextlib
import io
import logging
import time


def get_label_map():
    if cfg.dataset.label_map is None:
        return {x + 1: x + 1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map


def collate_fn_youtube_vis_eval(batch):
    return batch[0]


def collate_fn_youtube_vis(batch):
    # 0 imgs , 1 targets , 2 masks , 3 num_crowds
    frames = [([], [], [], []) for _ in batch[0][0]]   # TODO: is it better to use range here?
    for sample, extra in batch:
        for idx, (img, (gt, masks, num_crowds)) in enumerate(sample):
            frames[idx][0].append(img)
            frames[idx][1].append(torch.FloatTensor(gt) if gt is not None else gt)
            frames[idx][2].append(torch.FloatTensor(masks) if masks is not None else masks)
            frames[idx][3].append(num_crowds)

    for idx, (imgs, targets, masks, num_crowds) in enumerate(frames):
        frames[idx] = (torch.stack(imgs, 0),
                       (targets, masks, num_crowds), )

    return frames


class YoutubeVISAnnotationTransform(object):
    """Transforms a YoutubeVIS annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.dataset_name = cfg.dataset.name
        self.label_map = get_label_map()

    def __call__(self, target, frame_id, width, height):
        """
        Args:
            target (dict): YoutubeVIS target json annotation as a python dict
            frame_id (int): frame ID
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        # TODO: is this wasteful to check this? the dataset has been changed here.
        if self.dataset_name != cfg.dataset.name:
            self.label_map = get_label_map()
            self.dataset_name = cfg.dataset.name

        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bboxes' in obj and obj['bboxes'][frame_id] is not None:
                bbox = obj['bboxes'][frame_id]

                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1

                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            # else:
            # TODO: it shall be okay for videos to have some frames without bbox annotation, right?
            #     print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class YoutubeVIS(data.Dataset):
    """`YoutubeVIS <https://youtube-vos.org/dataset/vis/>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, configs, transform=None,
                 target_transform=YoutubeVISAnnotationTransform(),
                 dataset_name='YouTube VIS', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.ytvos import YTVOS

        self.root = image_path
        self.configs = configs

        logger = logging.getLogger("yolact.dataset")
        logger.info('Loading annotations into memory...')
        tic = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = YTVOS(info_file)

        self.ids = list(self.coco.vidToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.vids.keys())

        logger.info('{} videos loaded in {:0.2f}s.'.format(len(self.ids), time.time() - tic))

        self.transform = transform
        self.target_transform = target_transform

        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        video_frames, extra_data = self.pull_video(index)
        video_frames = [(im, (gt, masks, num_crowds), ) for im, gt, masks, h, w, num_crowds in video_frames]
        return video_frames, extra_data

    def pull_video(self, index, return_on_failure=False, full_video=False, max_images=-1):

        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        vid_id = self.ids[index]

        seq_len = self.configs.images_per_video

        # sample vid_id with enough length
        while True:
            vid = self.coco.loadVids(vid_id)[0]
            annot_length = len(vid['file_names'])
            if not full_video and annot_length < seq_len: continue  # FIXME: need to set new vid_id right?
            vid_name = vid['file_names'][0].split('/')[0]

            # Generate target starts.
            if self.has_gt:
                target = self.coco.vidToAnns[vid_id]
                ann_ids = self.coco.getAnnIds(vidIds=vid_id)

                # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bboxes', 'category_id'}
                target = self.coco.loadAnns(ann_ids)
            else:
                target = []

            # Separate out crowd annotations. These are annotations that signify a large crowd of
            # objects of said class, where there is no annotation for each individual object. Both
            # during testing and training, consider these crowds as neutral.
            crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
            target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
            num_crowds = len(crowd)

            for x in crowd:
                x['category_id'] = -1

            # This is so we ensure that all crowd annotations are at the end of the array
            target += crowd
            # Generate target ends.

            # shuffling and sample a small range of video here
            if full_video:
                annot_idx = np.arange(0, annot_length, 1)
                frame_idx = np.asarray([int(vid['file_names'][idx][-9:-4]) for idx in range(annot_length)])
                if self.configs.use_all_frames:
                    key_frame_idx = frame_idx
                    frame_idx = np.arange(frame_idx[0], frame_idx[-1] + 1, 1)
                    have_annot = np.asarray([int(idx in key_frame_idx) for idx in frame_idx])
                    annot_idx = np.add.accumulate(have_annot) * have_annot - 1

                if max_images != -1:
                    eval_frames = min(max_images, len(frame_idx))
                    # start_idx = np.random.randint(0, len(frame_idx) - eval_frames + 1)
                    start_idx = 0
                    frame_idx = frame_idx[start_idx: start_idx + eval_frames]
                    annot_idx = annot_idx[start_idx: start_idx + eval_frames]
            elif self.configs.use_all_frames:
                rand_idx = np.arange(0, annot_length - seq_len)
                np.random.shuffle(rand_idx)

                direction = 1
                if self.configs.all_frame_direction == 'allway':
                    if np.random.rand() > 0.5: direction *= -1
                elif self.configs.all_frame_direction == 'forward':
                    # Note: forward warping needs to sample a 'previous frame'
                    direction *= -1
                elif self.configs.all_frame_direction == 'backward':
                    pass
                else:
                    raise ValueError("Unexpected frame direction: %s" % self.configs.all_frame_direction)

                start_idx = rand_idx[0]
                if direction < 0:
                    start_idx += self.configs.images_per_video
                start_frame_idx = int(vid['file_names'][start_idx][-9:-4])
                annot_idx = [start_idx]
                frame_idx = [start_frame_idx]

                # if self.configs.images_per_video > 1:
                #     num_extra_frames = self.configs.images_per_video - 1
                #     extra_annot_idx = [start_idx + direction * offset_idx
                #                        for offset_idx in range(1, num_extra_frames + 1)]
                #     extra_frame_idx = [int(vid['file_names'][extra_idx][-9:-4])
                #                        for extra_idx in extra_annot_idx]
                #
                #     annot_idx += extra_annot_idx
                #     frame_idx += extra_frame_idx

                extra_frame_idx = []
                extra_annot_idx = []
                if self.configs.images_per_video > 0:
                    offset_lb, offset_ub = self.configs.frame_offset_lb, self.configs.frame_offset_ub
                    lb, ub = int(vid['file_names'][0][-9:-4]), int(vid['file_names'][-1][-9:-4])
                    fidx = frame_idx[-1]
                    lb, ub = lb - fidx, ub - fidx
                    if direction == -1:
                        ub = -offset_lb
                        lb = max(lb, -offset_ub)
                    else:
                        lb = offset_lb
                        ub = min(ub, offset_ub)
                    assert lb <= ub + 1, "{}, {}".format(lb, ub)
                    assert self.configs.frame_offset_multiplier == 1, "frame_offset_multiplier deprecated."
                    for _ in range(self.configs.images_per_video):
                        frame_diff = np.random.randint(lb, ub + 1)
                        ref_idx = fidx + frame_diff
                        assert int(vid['file_names'][0][-9:-4]) <= ref_idx <= int(vid['file_names'][-1][-9:-4]), "{} <= {} <= {}".format(int(vid['file_names'][0][-9:-4]), ref_idx, int(vid['file_names'][-1][-9:-4]))
                        # frame_diff = self.configs.frame_offset_multiplier * np.random.randint(self.configs.frame_offset_lb, self.configs.frame_offset_ub + 1)
                        # ref_idx = np.clip(frame_idx[-1] + frame_diff * direction,
                        #                   int(vid['file_names'][0][-9:-4]), int(vid['file_names'][-1][-9:-4]))
                        extra_frame_idx += [ref_idx]
                        extra_annot_idx += [-1]

                extra_frame_idx = list(sorted(extra_frame_idx, reverse=True))

                annot_idx += extra_annot_idx
                frame_idx += extra_frame_idx
                annot_idx = np.asarray(annot_idx)
                frame_idx = np.asarray(frame_idx)
            else:
                rand_idx = np.arange(0, annot_length - seq_len + 1)
                np.random.shuffle(rand_idx)
                start_idx = rand_idx[0]

                annot_idx = np.arange(start_idx, start_idx + seq_len, 1)
                frame_idx = np.asarray([int(vid['file_names'][idx][-9:-4]) for idx in annot_idx])

            has_targets = all([self.target_in_frame(target, annot_id, true_on_reference=True)
                               for annot_id in annot_idx])
            if has_targets: break
            if return_on_failure: return None
            # print("Not all frame of video %s[%d-%d] has targets, re-selecting video." %
            #       (vid['file_names'][0].split('/')[0], start_idx, start_idx + frm_len))
            index = np.random.randint(len(self))
            vid_id = self.ids[index]

        frame_results = []
        extra_data = []

        while True:
            try:
                for idx, (frame_id, annot_id) in enumerate(zip(frame_idx.tolist(), annot_idx.tolist())):
                    extra = {}
                    # FIXME: little bit hacky for full frames, maybe fix this using annotation files
                    frame_id_str = "%05d" % frame_id
                    file_name = vid['file_names'][0]
                    file_name = file_name[:-9] + frame_id_str + file_name[-4:]
                    prev_frame_id = frame_idx[idx - 1] if idx > 0 else -1
                    prev_annot_id = annot_idx[idx - 1] if idx > 0 else -1
                    if idx == 0:
                        seeds, (im, gt, masks, h, w, num_crowds) = self.pull_frame(vid_name, (frame_id, annot_id),
                                                                                         (prev_frame_id, prev_annot_id),
                                                                                         file_name,
                                                                                         target, num_crowds,
                                                                                         require_seeds=True)
                    else:
                        im, gt, masks, h, w, num_crowds = self.pull_frame(vid_name, (frame_id, annot_id),
                                                                                (prev_frame_id, prev_annot_id),
                                                                                file_name,
                                                                                target, num_crowds, seeds=seeds)

                    extra['idx'] = (frame_id, annot_id, )
                    frame_results.append((im, gt, masks, h, w, num_crowds, ))
                    extra_data.append(extra)
            except ValueError as e:
                logger = logging.getLogger("yolact.dataset")
                logger.warning('Resampling with reseed signal...')
                frame_results.clear()
                extra_data.clear()
                continue
            break

        return frame_results, extra_data

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def target_in_frame(target, frame_id, true_on_reference=False):
        if frame_id < 0:
            return true_on_reference
        if len(target) > 0:
            for obj in target:
                if obj['segmentations'][frame_id] is not None:
                    return True
        return False

    def pull_frame(self, vid_name, frame_annot_id, prev_frame_annot_id, file_name, target, num_crowds, require_seeds=False, seeds=None):
        frame_id, annot_id = frame_annot_id
        prev_frame_id, prev_annot_id = prev_frame_annot_id
        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        target_is_in_frame = self.target_in_frame(target, annot_id)

        if target_is_in_frame:
            # Pool all the masks for this image into one [num_objects,height,width] matrix

            # masks = [np.zeros(height * width, dtype=np.uint8).reshape(-1) if obj['segmentations'][frame_id] is None  # all-zero mask on None
            #          else self.coco.annToMask(obj, frame_id).reshape(-1) for obj in target]
            masks = [self.coco.annToMask(obj, annot_id).reshape(-1)
                     for obj in target
                     if obj['segmentations'][annot_id] is not None]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and target_is_in_frame:
            target = self.target_transform(target, annot_id, width, height)

        if self.transform is not None:
            if "Video" in type(self.transform).__name__:
                if target_is_in_frame:
                    target = np.array(target)
                    return_transform = self.transform(img, masks, target[:, :4],
                                                      {'num_crowds': num_crowds, 'labels': target[:, 4]},
                                                      require_seeds=require_seeds, seeds=seeds)

                    if require_seeds:
                        seeds, (img, masks, boxes, labels) = return_transform
                    else:
                        img, masks, boxes, labels = return_transform

                    # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                    num_crowds = labels['num_crowds']
                    labels = labels['labels']

                    target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

                    if target.shape[0] == 0:
                        logger = logging.getLogger("yolact.dataset")
                        logger.warning('Augmentation output an example with no ground truth. Resampling...')
                        raise ValueError("reseed")
                else:
                    try:
                        return_transform = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                                  np.array([[0., 0., 1., 1.]]),
                                                  {'num_crowds': 0, 'labels': np.array([0])},
                                                      require_seeds=require_seeds, seeds=seeds)
                    except ValueError:
                        assert False, "Unexpected reseed captured with no-target instances."

                    if require_seeds:
                        seeds, (img, _, _, _) = return_transform
                    else:
                        img, _, _, _ = return_transform

                    masks = None
                    target = None
            else:
                if target_is_in_frame:
                    target = np.array(target)
                    img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                               {'num_crowds': num_crowds, 'labels': target[:, 4]})

                    # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                    num_crowds = labels['num_crowds']
                    labels = labels['labels']

                    target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                else:
                    img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                                  np.array([[0, 0, 1, 1]]),
                                                  {'num_crowds': 0, 'labels': np.array([0])})
                    masks = None
                    target = None

        return_tuple = torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds
        if require_seeds:
            return seeds, return_tuple
        else:
            return return_tuple

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class YoutubeVISEval(data.Dataset):
    def __init__(self, dataset, indices, max_images):
        self.dataset = dataset
        self.indices = indices
        self.max_images = max_images


    def __getitem__(self, idx):
        video_idx = self.indices[idx]
        return self.dataset.pull_video(video_idx, return_on_failure=True, full_video=True, max_images=self.max_images)

    def __len__(self):
        return len(self.indices)

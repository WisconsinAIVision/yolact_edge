import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from yolact_edge.data import cfg, MEANS, STD


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None):
        for t in self.transforms:
            img, masks, boxes, labels = t(img, masks, boxes, labels)
        return img, masks, boxes, labels


class ComposeVideo(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        new_seeds = []
        for idx, t in enumerate(self.transforms):
            if require_seeds:
                new_seed, (img, masks, boxes, labels) = t(img, masks, boxes, labels, seeds=None,
                                                                 require_seeds=True)
                new_seeds.append(new_seed)
            else:
                img, masks, boxes, labels = t(img, masks, boxes, labels, seeds=seeds[idx],
                                                     require_seeds=False)
        if require_seeds:
            return new_seeds, (img, masks, boxes, labels)
        return img, masks, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.lambd(img, masks, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if require_seeds:
            return None, (image.astype(np.float32), masks, boxes, labels)
        else:
            return image.astype(np.float32), masks, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        if require_seeds:
            return None, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        if require_seeds:
            return None, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, width, height, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width = width
        self.height = height
        self.pad_gt = pad_gt

        if type(width) == tuple:
            assert width == height
            self.width, self.height = height

    def __call__(self, image, masks, boxes=None, labels=None, seeds=None, require_seeds=False):
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:,:im_h,:im_w] = masks
            masks = expand_masks


        if require_seeds:
            return None, (expand_image, masks, boxes, labels)
        else:
            return expand_image, masks, boxes, labels


class Resize(object):
    """
    The same resizing scheme as used in faster R-CNN
    https://arxiv.org/pdf/1506.01497.pdf

    We resize the image so that the shorter side is min_size.
    If the longer side is then over max_size, we instead resize
    the image so the long side is max_size.
    """

    @staticmethod
    def faster_rcnn_scale(width, height, min_size, max_size):
        min_scale = min_size / min(width, height)
        width  *= min_scale
        height *= min_scale

        max_scale = max_size / max(width, height)
        if max_scale < 1: # If a size is greater than max_size
            width  *= max_scale
            height *= max_scale
        
        return int(width), int(height)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.min_size = cfg.min_size
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None, seeds=None, require_seeds=False):
        img_h, img_w, _ = image.shape
        
        if self.preserve_aspect_ratio:
            width, height = Resize.faster_rcnn_scale(img_w, img_h, self.min_size, self.max_size)
        elif type(self.max_size) == tuple:
            width, height = self.max_size
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))
        
        if self.resize_gt:
            # Act like each object is a color channel
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))
            
            # OpenCV resizes a (w,h,1) array to (s,s), so fix that
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            # Scale bounding boxes (which are currently absolute coordinates)
            boxes[:, [0, 2]] *= (width  / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        if require_seeds:
            return None, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if seeds is None:
            if random.randint(2):
                seeds = random.uniform(self.lower, self.upper)
            else:
                seeds = 1.0
        image[:, :, 1] *= seeds

        if require_seeds:
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if seeds is None:
            if random.randint(2):
                seeds = random.uniform(-self.delta, self.delta)
            else:
                seeds = 0

        image[:, :, 0] += seeds
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        if require_seeds:
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)

        if require_seeds:
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        if require_seeds:
            return None, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if seeds is None:
            if random.randint(2):
                seeds = random.uniform(self.lower, self.upper)
            else:
                seeds = 1.0
        alpha = seeds
        image *= alpha

        if require_seeds:
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        if seeds is None:
            if random.randint(2):
                seeds = random.uniform(-self.delta, self.delta)
            else:
                seeds = 0

        delta = seeds
        image += delta

        if require_seeds:
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, masks, boxes=None, labels=None, seeds=None, require_seeds=False):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            if seeds is None:
                mode = random.choice(self.sample_options)
            else:
                mode = seeds[0]

            if mode is None:
                if require_seeds:
                    seeds = (mode, )
                    return seeds, (image, masks, boxes, labels)
                else:
                    return image, masks, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                if seeds is None:
                    w = random.uniform(0.3 * width, width)
                    h = random.uniform(0.3 * height, height)
                else:
                    w, h = seeds[1:3]

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    if seeds is not None:
                        raise ValueError("reseed")
                    continue

                if seeds is None:
                    left = random.uniform(width - w)
                    top = random.uniform(height - h)
                else:
                    left, top = seeds[3:5]

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    if seeds is not None:
                        raise ValueError("reseed")
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1-crowd_mask[mask]) == 0:
                    if seeds is not None:
                        if not masks.any():
                            return current_image, masks, boxes, labels
                        raise ValueError("reseed")
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                if require_seeds:
                    seeds = (mode, w, h, left, top)
                    return seeds, (current_image, current_masks, current_boxes, current_labels)
                else:
                    return current_image, current_masks, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, labels, seeds=None, require_seeds=False):
        if seeds is not None:
            random_draw = seeds[0]
        else:
            random_draw = random.randint(2)

        if random_draw:
            if require_seeds:
                seeds = (random_draw, )
                return seeds, (image, masks, boxes, labels)
            else:
                return image, masks, boxes, labels

        height, width, depth = image.shape

        if seeds is not None:
            ratio, left, top = seeds[1:4]
        else:
            ratio = random.uniform(1, 4)
            left = random.uniform(0, width*ratio - width)
            top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros(
            (masks.shape[0], int(height*ratio), int(width*ratio)),
            dtype=masks.dtype)
        expand_masks[:,int(top):int(top + height),
                       int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        if require_seeds:
            seeds = (random_draw, ratio, left, top)
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomMirror(object):
    def __call__(self, image, masks, boxes, labels, seeds=None, require_seeds=False):
        _, width, _ = image.shape

        if seeds is not None:
            random_draw = seeds[0]
        else:
            random_draw = random.randint(2)

        if random_draw:
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        if require_seeds:
            seeds = (random_draw, )
            return seeds, (image, masks, boxes, labels)
        else:
            return image, masks, boxes, labels


class RandomFlip(object):
    def __call__(self, image, masks, boxes, labels, seeds=None, require_seeds=False):
        height , _ , _ = image.shape

        if seeds is not None:
            random_draw = seeds[0]
        else:
            random_draw = random.randint(2)

        if random_draw:
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]

        if require_seeds:
            seeds = (random_draw, )
            return image, masks, boxes, labels
        else:
            return image, masks, boxes, labels

class RandomRot90(object):
    def __call__(self, image, masks, boxes, labels, seeds=None, require_seeds=False):
        old_height , old_width , _ = image.shape

        if seeds is not None:
            random_draw = seeds[0]
        else:
            random_draw = random.randint(4)

        k = random_draw
        image = np.rot90(image,k)
        masks = np.array([np.rot90(mask,k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            old_width, old_height = old_height, old_width

        if require_seeds:
            seeds = (random_draw, )
            return seeds, (image, masks, boxes, labels)

        return image, masks, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, labels, seeds=None, require_seeds=False):
        im = image.copy()
        if seeds is None:
            brightness_seed, (im, masks, boxes, labels) = self.rand_brightness(im, masks, boxes, labels, require_seeds=True)
            distort_seed_1 = random.randint(2)
            if distort_seed_1:
                distort = ComposeVideo(self.pd[:-1])
            else:
                distort = ComposeVideo(self.pd[1:])
            distort_seed, (im, masks, boxes, labels) = distort(im, masks, boxes, labels, require_seeds=True)
            seeds = (brightness_seed, distort_seed_1, distort_seed)
            im, masks, boxes, labels = self.rand_light_noise(im, masks, boxes, labels)
            if require_seeds:
                return seeds, (im, masks, boxes, labels)
            else:
                return im, masks, boxes, labels
        else:
            brightness_seed, distort_seed_1, distort_seed = seeds
            im, masks, boxes, labels = self.rand_brightness(im, masks, boxes, labels, seeds=brightness_seed)
            if distort_seed_1:
                distort = ComposeVideo(self.pd[:-1])
            else:
                distort = ComposeVideo(self.pd[1:])
            im, masks, boxes, labels = distort(im, masks, boxes, labels, seeds=distort_seed)
            im, masks, boxes, labels = self.rand_light_noise(im, masks, boxes, labels)
            
            return im, masks, boxes, labels


class PrepareMasks(object):
    """
    Prepares the gt masks for use_gt_bboxes by cropping with the gt box
    and downsampling the resulting mask to mask_size, mask_size. This
    function doesn't do anything if cfg.use_gt_bboxes is False.
    """

    def __init__(self, mask_size, use_gt_bboxes):
        self.mask_size = mask_size
        self.use_gt_bboxes = use_gt_bboxes

    def __call__(self, image, masks, boxes, labels=None):
        if not self.use_gt_bboxes:
            return image, masks, boxes, labels
        
        height, width, _ = image.shape

        new_masks = np.zeros((masks.shape[0], self.mask_size ** 2))

        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes[i, :]
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))

            # +1 So that if y1=10.6 and y2=10.9 we still have a bounding box
            cropped_mask = masks[i, y1:(y2+1), x1:(x2+1)]
            scaled_mask = cv2.resize(cropped_mask, (self.mask_size, self.mask_size))

            new_masks[i, :] = scaled_mask.reshape(1, -1)
        
        # Binarize
        new_masks[new_masks >  0.5] = 1
        new_masks[new_masks <= 0.5] = 0

        return image, new_masks, boxes, labels

class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self, transform, mean, std, in_channel_order):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, masks=None, boxes=None, labels=None):

        img = img.astype(np.float32)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        img = img[:, :, self.channel_permutation]

        return img.astype(np.float32), masks, boxes, labels




class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            Resize(resize_gt=False),
            Pad(cfg.max_size, cfg.max_size, mean, pad_gt=False),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


class BaseTransformVideo(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment_s1 = ComposeVideo([
            ConvertFromInts(),
            Resize(resize_gt=False),
            Pad(cfg.max_size, cfg.max_size, mean, pad_gt=False)
        ])
        self.augment_s2 = Compose([
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
        return_batch = self.augment_s1(img, masks, boxes, labels, seeds=seeds, require_seeds=require_seeds)
        if require_seeds:
            seeds, return_batch = return_batch
        img, masks, boxes, labels = return_batch
        img, masks, boxes, labels = self.augment_s2(img, masks, boxes, labels)

        return_batch = img, masks, boxes, labels
        if require_seeds:
            return seeds, return_batch
        else:
            return return_batch


import torch.nn.functional as F

class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        if cfg.preserve_aspect_ratio:
            raise NotImplementedError

        img = img.permute(0, 3, 1, 2).contiguous()
        if type(cfg.max_size) == tuple:
            img = F.interpolate(img, cfg.max_size[::-1], mode='bilinear', align_corners=False)
        else:
            img = F.interpolate(img, (cfg.max_size, cfg.max_size), mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img

def do_nothing(img=None, masks=None, boxes=None, labels=None, seeds=None, require_seeds=False):
    if require_seeds:
        return None, (img, masks, boxes, labels)
    else:
        return img, masks, boxes, labels


def enable_if(condition, obj):
    return obj if condition else do_nothing

class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment_expand, Expand(mean)),
            enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.augment_random_mirror, RandomMirror()),
            enable_if(cfg.augment_random_flip, RandomFlip()),
            enable_if(cfg.augment_random_flip, RandomRot90()),
            Resize(),
            Pad(cfg.max_size, cfg.max_size, mean),
            ToPercentCoords(),
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks, boxes, labels):
        return self.augment(img, masks, boxes, labels)


class SSDAugmentationVideo(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment_s1 = ComposeVideo([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment_expand, Expand(mean)),
            enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.augment_random_mirror, RandomMirror()),
            enable_if(cfg.augment_random_flip, RandomFlip()),
            enable_if(cfg.augment_random_flip, RandomRot90()),
            Resize(),
            Pad(cfg.max_size, cfg.max_size, mean),
            ToPercentCoords(),
        ])
        self.augment_s2 = Compose([
            PrepareMasks(cfg.mask_size, cfg.use_gt_bboxes),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR')
        ])

    def __call__(self, img, masks, boxes, labels, seeds=None, require_seeds=False):
        return_batch = self.augment_s1(img, masks, boxes, labels, seeds=seeds, require_seeds=require_seeds)
        if require_seeds:
            seeds, return_batch = return_batch

        img, masks, boxes, labels = return_batch
        img, masks, boxes, labels = self.augment_s2(img, masks, boxes, labels)

        return_batch = img, masks, boxes, labels
        if require_seeds:
            return seeds, return_batch
        else:
            return return_batch

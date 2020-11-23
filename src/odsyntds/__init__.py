from importlib.metadata import version

__version__ = version(__package__)

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

import albumentations
from albumentations import Compose
from albumentations.core.composition import BboxParams
import cv2
import imutils

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def overlay(objects, bkg):
    """
    overlay a source image containing bounding boxes onto a background

    Parameters:
    -----
    objects : list 
        list of the source image(s), potentially augmented (rotated, scaled, ...) with their associated data
    bkg : numpy array
        the background on which the objects will be overlayed

    Returns:
    --------
    bkg_with_objects : numpy array
        the background with overlayed objects
    target_bboxes : list 
        list of imgaug.augmentables.bbs.BoundingBoxesOnImage : the bounding boxes of the sources images in the background image coordinate reference frame

    Details about the overlay function
    -------------------
    The overlay function works in conjunction with the imgaug package.
    Its ojective is to allow the user to overlay an object represented by a source
    image and an associated bounding box into a bigger picture called the background
    it takes as input a list of cv2 images containing (or not) an alpha channel,
    a list of corresponding bounding boxes and a background.

    The script then ensures that the overlayed source will have adequate proportions
    when overlayed into the background i.e. it restricts the overlayed source
    to be between 1% and 5% of the background x-dimension. The y dimension is
    then scaled accordingly to keep the aspect ratio of the original source image.
    The associated bounding box are recomputed accordingly.

    The script overlays the source randomly into the background and allows the
    overlayed source to sit on the edge of the background i.e. allows for an
    incomplete overlay. The bounding box associated to this overlayed source
    are only kept if more than 25% of the source sits within the background.
    Note that only the bounding box is removed, not the overlayed source i.e.
    the overlayed source will appear in the picture but no bounding box will be
    associated to it.
    """
    bkg_with_objects = {key: None for key in ["image", "bboxes", "class_id"]}

    if bkg.shape[2] != 4:
        bkg = cv2.cvtColor(bkg, cv2.COLOR_RGB2RGBA)

    for object in objects:

        # reshape source in order to limit its size on the bkg
        # ratio is chosen based on the width i.e. x-axis
        # the figures are based on analysis performed on the desired
        # detections
        ratioChoice = np.arange(0.2, 0.3, step=0.01)
        ratioChosen = np.random.choice(ratioChoice, 1)[0]

        resize = albumentations.augmentations.transforms.LongestMaxSize(
            max_size=int(ratioChosen * bkg.shape[1]), always_apply=True
        )
        aug = Compose(
            [resize],
            bbox_params=BboxParams(
                format="yolo",
                min_visibility=0.5,
                label_fields=["class_id"],
            ),
        )

        object = aug(**object)

        # create a padded version of the background
        # allowing the source picture to sit on the edges i.e.
        # to be partially outside the picture
        # the padding equal to the source width and height i.e.
        # actually allows for the source to be located completely outside
        # the background
        hO = object["image"].shape[0]
        wO = object["image"].shape[1]
        hB = bkg.shape[0]
        wB = bkg.shape[1]
        bkg_padded = cv2.copyMakeBorder(
            bkg,
            hO,
            hO,
            wO,
            wO,
            cv2.BORDER_CONSTANT,
        )

        # Anchor is chosen randomly within the padded image
        # but without allowing the picture to lie outside the original image
        # in other words, the padding is not used at present
        # padding shall be used in a further step to allow partial image to be
        # detected (this supposedly would help training) but this requires
        # to recompute bounding boxes after inclusion (otherwhise, center of
        # bbox could be outside the picture !)
        easy = False
        if easy:
            xAnchor = round(bkg_padded.shape[1] / 2)
            yAnchor = round(bkg_padded.shape[0] / 2)
        else:
            xAnchor = np.random.choice(
                np.arange(
                    round(1.5 * wO),
                    bkg_padded.shape[1] - round(1.5 * wO),
                )
            )  # anchor represents the center. Therefore, only allowed between these bounds in the padded image
            yAnchor = np.random.choice(
                np.arange(
                    round(1.5 * hO),
                    bkg_padded.shape[0] - round(1.5 * hO),
                )
            )
        # note the round hereabove are for future when widthSource/2 will have to be used
        # to allow partial inclusion of bbox

        # compute the boundaries of the source image inside the total image
        # This is performed assuming that in the source image, the
        # object center is perfectly in the middle of the source image &
        # associated bounding box
        # Therefore, one only to add/remove half the height and width with respect
        # to the chosen anchor to find the two opposite corners of the
        # source/bounding box inside the padded background
        xLeft = int(round(xAnchor - wO / 2))
        xRight = int(round(xAnchor + wO / 2))
        yBot = int(round(yAnchor - hO / 2))
        yTop = int(round(yAnchor + hO / 2))

        # can happen due to roundings that the shape of the image in the
        # computed location exceeds/is smaller than original image
        # correct that
        if (xRight - xLeft > wO) | (xRight - xLeft < wO):
            xLeft = xRight - wO
        if (yTop - yBot > hO) | (yTop - yBot < hO):
            yBot = yTop - hO

        # can also happen that when recomputing the location,
        # the coordinates go negative which is impossible
        # therefore, make coordinate positive and then
        # translate by an amount equal to the negative value of the
        # coordinate
        if xLeft < 0:
            xLeft = abs(xLeft)
            delta = 2 * xLeft
            xRight = xRight + delta
        if yBot < 0:
            yBot = abs(yBot)
            delta = 2 * yBot
            yTop = yTop + delta

        # perform overlay at chosen location
        # takes into account alpha channel of the source
        alpha = object["image"][:, :, -1]
        alpha_rgb = np.expand_dims(alpha, 2)
        alpha_rgb = np.repeat(alpha_rgb, 3, axis=2)
        alpha_rgb = alpha_rgb.astype(float) / 255
        foreground = cv2.multiply(
            alpha_rgb, object["image"][:, :, :-1], dtype=cv2.CV_64F
        )
        background = cv2.multiply(
            1 - alpha_rgb,
            bkg_padded[yBot:yTop, xLeft:xRight][:, :, :-1],
            dtype=cv2.CV_32F,
        )
        bkg_padded[yBot:yTop, xLeft:xRight][:, :, :-1] = cv2.add(
            foreground, background, dtype=cv2.CV_64F
        )
        # bkg_padded[yBot:yTop, xLeft:xRight][:, :, -1] = alpha # WRONG
        # Keep only the original picture without the padding
        bkg = bkg_padded[
            hO : hO + bkg.shape[0],
            wO : wO + bkg.shape[1],
        ]
        # In source image, bounding box is around the entire image
        # and assumed at the middle therefore one computes the distance between
        # the center of the bounding box and its edges
        # note the convention in imgAug : x-axis grows from left to right
        # y-axis grows from top to bottom
        # xCtr = wO/2
        # yCtr = hO/2
        # deltaX = abs(sourceBbox.bounding_boxes[0].x1_int - xCtr)
        # deltaY = abs(sourceBbox.bounding_boxes[0].y1_int - yCtr)

        # Now define the new bounding box associated to the overlayed source
        # inside the background image. The associated bboxes will be placed
        # in the original (non-padded) background referential so
        # define the bbox coordinates consistently i.e. removing
        # widthSource and heightSource from the coordinates
        # make a drawing to vizualize if needed for understanding
        bkg_with_objects["image"] = bkg
        try:
            bkg_with_objects["bboxes"].append(
                [(xAnchor - wO) / wB, (yAnchor - hO) / hB, wO / wB, hO / hB]
            )
            bkg_with_objects["class_id"].append(object["class_id"][0])
        except AttributeError:
            bkg_with_objects["bboxes"] = [
                [(xAnchor - wO) / wB, (yAnchor - hO) / hB, wO / wB, hO / hB]
            ]
            bkg_with_objects["class_id"] = [object["class_id"][0]]

    # # view
    # # imgaug.imshow(BoundingBoxesOnImage(targetBboxes, bkg.shape).draw_on_image(bkg, color = (0,255,0,255)))
    # # check if bounding boxes are within the picture
    # i = 0
    # for bbox in targetBboxes:
    #     originalArea = bbox.area
    #     remainingArea = bbox.clip_out_of_image(
    #         bkg.shape
    #     ).area  # this computes the remaining area when cut
    #     if remainingArea / originalArea < 0.4:
    #         targetBboxes.pop(i)
    #         # don't increment i if popping element
    #     else:
    #         targetBboxes[i] = bbox.clip_out_of_image(bkg.shape)
    #         i += 1

    return bkg_with_objects


BOX_COLOR = (255, 0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def draw_bbox(annotations, class_id_to_name):

    img = annotations["image"].copy()
    color = BOX_COLOR
    thickness = 2

    for idx, bbox in enumerate(annotations["bboxes"]):
        class_id = annotations["class_id"][idx]
        xc_bbox_abs, yc_bbox_abs, w_bbox_abs, h_bbox_abs = (
            bbox[0] * img.shape[1],
            bbox[1] * img.shape[0],
            bbox[2] * img.shape[1],
            bbox[3] * img.shape[0],
        )
        x_min, y_min, x_max, y_max = (
            int(xc_bbox_abs - w_bbox_abs / 2),
            int(yc_bbox_abs - h_bbox_abs / 2),
            int(xc_bbox_abs + w_bbox_abs / 2),
            int(yc_bbox_abs + h_bbox_abs / 2),
        )
        cv2.rectangle(
            img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness
        )
        class_name = class_id_to_name[class_id]
        ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        cv2.rectangle(
            img,
            (x_min, y_min - int(1.3 * text_height)),
            (x_min + text_width, y_min),
            BOX_COLOR,
            -1,
        )
        cv2.putText(
            img,
            class_name,
            (x_min, y_min - int(0.3 * text_height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img


def visualize(img_with_bbox):
    plt.figure(figsize=(12, 12))
    plt.imshow(img_with_bbox)
    plt.show()


def makeObjects(obj_path):
    """
    Create the objects to be detected

    Parameters
    ----------
    obj_path : pathlib.Path
        directory containing the objects

    Returns
    -------
    a list of objects (dictionnaries) and a dict containing the mapping from id to object classes

    Notes
    -----
    Each object is a dictionnary containing an image, bboxes (in YOLO format) and class_id keys.
    The objects' classes are derived from the filenames e.g. valve.jpg will be assigned the class 'valve'.

    YOLO format : [xcenter, ycenter, width, height] all relative to total picture size

    """
    handled_extensions = ["png", "jpg"]
    objects = []
    ids_to_classes = {}

    for i, object in enumerate(obj_path.iterdir()):
        if object.suffix in handled_extensions:
            ids_to_classes[i] = object.stem

            object = DefaultDict()

            object["image"] = cv2.imread(
                str(object), cv2.IMREAD_UNCHANGED
            )  # Keep the alpha (transparancy) channel

            h = object["image"].shape[0]
            w = object["image"].shape[1]
            bboxMargin = (
                2  # in px, 1px is minimum otherwhise albumentation complains when augmenting
                # 2px avoids rounding issues (greater than 1.0 for box size, ...)
            )
            hbbox = h - 2 * bboxMargin  # bbox takes whole image minus margin
            wbbox = w - 2 * bboxMargin
            object["bboxes"] = [
                [
                    0.5,
                    0.5,
                    wbbox / w,
                    hbbox / h,
                ]
            ]  # yolo format [xcenter, ycenter, width, height] all relative to total picture size

            object["class_id"] = [id]

            objects.append(object)
        else:
            logger.info(
                "file {object} has been ignored because its extension"
                "does not belong to one of the handled extensions :{}"
            ).format(object, ",".join(handled_extensions))
    # TODELETE object = {key: None for key in ["image", "bboxes", "class_id"]}

    return objects, ids_to_classes


def rbga2rbg(img):
    trans_mask = img[:, :, 3] == 0
    img[trans_mask] = [255, 255, 255, 255]
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


def resizeYoloV3(img, dim):
    """
    Resize an image to a multiple of YOLOV3 stride (32 pixels)

    Parameters
    ----------
    img (cv2.) :
    dim : int
        the size in pixels to which the biggest dimension of the image will be resized
    """
    # resize only (to a multiple of YoloV3 stride (32))
    h, w = img.shape[:-1]
    maxDim = np.argmax([h, w])
    ar = h / w
    if maxDim == 0:  # if maxDim is axis 0, resize axis 0
        newSize = dim, round(dim / ar)  # maybe not divisible by 32 ...
        quo, rest = divmod(newSize[1], 32)
        # (w, h)
        newSize = (
            32 * quo,
            dim,
        )  # therefore make it divisible by 32 even if modifying aspect ratio (ar)
    else:  # if maxDim is axis 1, resize axis 1
        newSize = round(ar * dim), dim
        quo, rest = divmod(newSize[0], 32)
        # (w, h)
        newSize = (dim, 32 * quo)
    return cv2.resize(img, newSize)


def augment_objects(objects, per_obj):
    """
    Perform image augmentation on objects and their associated bboxes

    Parameters
    ----------
    objects : list
        a list of objects (dictionnaries)
    per_obj : int
        number of augmented objects to create per object

    Returns
    -------
    A list containing len(objects)*per_obj augmented objects


    """

    # schedule augmentation of overlies
    rotate = albumentations.augmentations.transforms.RandomRotate90(
        p=0.0
    )  # TODO : allow user to define augmentations (true e.g. config or yaml file)
    aug = Compose(
        [rotate],
        bbox_params=BboxParams(
            format="yolo",
            min_visibility=0.5,
            label_fields=["class_id"],
        ),
    )
    # perform augmentation on overlies
    # obj_per_bkg = max(len(objects), np.random.choice(1))
    aug_per_obj = int(round(per_obj / len(objects)))
    augmented = []
    for object in objects:
        temp = [aug(**object) for i in range(aug_per_obj)]
        augmented = augmented + temp

    return augmented


def write_annotations(fp, img_with_bboxes, format):
    """
    write annotations related to bboxes to a file

    fp : pathlib.Path
        the filepath of the annotation file
    img_with_bboxes : dict
        a dict containing the image and bboxes information
    format : str {'yolo', 'SSD'}
        the format for saving the bboxes coordinates
    """

    with open(
        fp,
        mode="w",
        encoding="utf-8",
        newline=None,
    ) as f:
        for i, bbox in enumerate(img_with_bboxe["bboxes"]):
            if format == "yolo":
                yolo = "{:d} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                    *[int(img_with_bboxes["classs_id"][i])] + bbox
                )
                f.write(yolo)
            elif format == "SSD":
                img_w = img_with_bboxes["image"].shape[1]
                img_h = img_with_bboxes["image"].shape[0]
                bbox_xcenter_rel = bbox[0]
                bbox_ycenter_rel = bbox[1]
                bbox_w_rel = bbox[2]
                bbox_h_rel = bbox[3]
                xmin_rel = bbox_xcenter_rel - bbox_w_rel / 2
                xmax_rel = bbox_xcenter_rel + bbox_w_rel / 2
                ymin_rel = bbox_ycenter_rel - bbox_h_rel / 2
                ymax_rel = bbox_ycenter_rel + bbox_h_rel / 2

                SSD = "{:d} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                    *[int(img_with_bboxes["classs_id"][i])]
                    + [xmin_rel, xmax_rel, ymin_rel, ymax_rel]
                )
                f.write(SSD)


def create_ds(
    obj_path,
    bkg_path,
    save_path='trainingSet',
    perBkg=10,
    format="yolo",
    aug=True,
    resize=None,
    no_alpha=False,
    save_bboxes=True,
):
    """
    Create a synthetic dataset for object detection tasks starting from a set of objects (to be detected) and a set of backgrounds

    Parameters
    ----------
    obj_path : str
        the path to the directory containing the objects.
    bkg_path : str
        the path to the directory containing the backgrounds.
    save_path : str
        the path to the directory containing the backgrounds.
    perBkg : int, optional
        the number of samples that will be created per background
    format : str, optional {'yolo', 'SSD'}
        the format for saving the bboxes coordinates
    resize : int, optional
        the size in pixels to which the biggest dimension of the background will be resized
    no_alpha : bool, optional
        whether to include the alpha channel or not in the synthetically generated dataset samples (images)
    save_bboxes : bool, optional
        wheter or not to save a copy of each sample showing the objects' bboxes

    Notes
    -----
    the directories should contain raster images (.png, .jpg) of the desired objects to be detected and the
    backgrounds to be used.
    Particularly, the objects' images files should have tight margins around the object i.e. no white pixels space around the object

    Also, the resize function will actually resize to the closest multiple of 32 pixels closest to the rescribed size.
    This is related to the fact one wants to have an image size which is a multiple of yolo stride.
    """
    handled_extensions = ["png", "jpg"]

    obj_path = pathlib(obj_path)
    bkg_path = Path(bkg_path)

    trainingSetPath = Path(args.save)
    if not trainingSetPath.exists():
        trainingSetPath.mkdir()

    objects, ids_to_classes = makeObjects(obj_path)

    counter = 0
    for background in backgroundPath.iterdir():
        if background.suffix in handled_extensions:
            logger.info("processing background {}".format(background))

            backgroundImg = cv2.imread(str(background), cv2.IMREAD_UNCHANGED)
            if resize:
                backgroundImg = resizeYoloV3(backgroundImg, resize)
            h, w = backgroundImg.shape[:-1]
            if h > w:  # make all background display as landscape
                backgroundImg = imutils.rotate_bound(backgroundImg, 90)

            # per background pass, create a new set of augmented overlies,
            # if requested by the user
            for j in range(perBkg):
                if aug:
                    objects_aug = augment_objects(objects)
                else:
                    objects_aug = objects

                # overlay objects on background
                bkg_with_obj = overlay(objects_aug, backgroundImg)

                if no_alpha:
                    bkg_with_obj["image"] = rbga2rbg(bkg_with_obj["image"])

                if save_bboxes:
                    bkg_with_bbox = draw_bbox(bkg_with_obj, ids_to_classes)
                    savePath = Path("trainingSet_with_bbox").joinpath(
                        "train_sample_" + str(counter) + ".png"
                    )
                    if not savePath.parent.exists():
                        savePath.parent.mkdir(parents=True)
                    cv2.imwrite(
                        str(savePath),
                        bkg_with_bbox,
                    )

                # save the sample
                cv2.imwrite(
                    str(
                        trainingSetPath.joinpath(
                            "train_sample_" + str(counter) + ".png"
                        )
                    ),
                    bkg_with_obj["image"],
                )

                write_annotations(
                    trainingSetPath.joinpath("train_sample_" + str(counter) + ".txt"),
                    bkg_with_obj,
                    format,
                )
                counter += 1
        else:
            logger.info(
                "file {} has been ignored because its extension"
                "does not belong to one of the handled extensions :{}"
            ).format(background, ",".join(handled_extensions))

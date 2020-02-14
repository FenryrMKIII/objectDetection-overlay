import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

import albumentations
from albumentations import BboxParams, Compose


def overlay(overlies, bckg):
    """
    overlay a source image containing bounding boxes onto a background
    
    Parameters:
    -----
    overlies(list of dictionnaries): the source image(s), potentially augmented (rotated, scaled, ...) with their associated data
    bckg (numpy array) : the background on which the overlies will be overlayed

    Returns:
    --------
    bckg (numpy array) : the background with overlayed overlies
    targetBboxes (list of imgaug.augmentables.bbs.BoundingBoxesOnImage) : the bounding boxes of the sources images in the background image coordinate reference frame

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
    bckgWithOverlies = {key: None for key in ["image", "bboxes", "category_id"]}

    if bckg.shape[2] != 4:
        bckg = cv2.cvtColor(bckg, cv2.COLOR_RGB2RGBA)

    for overly in overlies:

        # reshape source in order to limit its size on the bckg
        # ratio is chosen based on the width i.e. x-axis (arbitrarily)
        # ratioChoice = np.arange(0.2, 0.06, step=0.1)
        ratioChosen = 0.02  # np.random.choice(ratioChoice)

        resize = albumentations.augmentations.transforms.LongestMaxSize(
            max_size=int(ratioChosen * bckg.shape[1]), always_apply=True
        )
        aug = Compose(
            [resize],
            bbox_params=BboxParams(
                format="yolo",
                min_area=0,
                min_visibility=0.5,
                label_fields=["category_id"],
            ),
        )

        overly = aug(**overly)

        # create a padded version of the background
        # allowing the source picture to sit on the edges i.e.
        # to be partially outside the picture
        # the padding equal to the source width and height i.e.
        # actually allows for the source to be located completely outside
        # the background
        hO = overly["image"].shape[0]
        wO = overly["image"].shape[1]
        hB = bckg.shape[0]
        wB = bckg.shape[1]
        bckgPadded = cv2.copyMakeBorder(bckg, hO, hO, wO, wO, cv2.BORDER_CONSTANT,)

        # Anchor is chosen randomly within the padded image
        # but without allowing the picture to lie outside the original image
        # in other words, the padding is not used at present
        # padding shall be used in a further step to allow partial image to be
        # detected (this supposedly would help training) but this requires
        # to recompute bounding boxes after inclusion (otherwhise, center of
        # bbox could be outside the picture !)
        easy = False
        if easy:
            xAnchor = round(bckgPadded.shape[1] / 2)
            yAnchor = round(bckgPadded.shape[0] / 2)
        else:
            xAnchor = np.random.choice(
                np.arange(round(1.5 * wO), bckgPadded.shape[1] - round(1.5 * wO),)
            )  # anchor represents the center. Therefore, only allowed between these bounds in the padded image
            yAnchor = np.random.choice(
                np.arange(round(1.5 * hO), bckgPadded.shape[0] - round(1.5 * hO),)
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
        alpha = overly["image"][:, :, -1]
        alpha_rgb = np.expand_dims(alpha, 2)
        alpha_rgb = np.repeat(alpha_rgb, 3, axis=2)
        alpha_rgb = alpha_rgb.astype(float) / 255
        foreground = cv2.multiply(
            alpha_rgb, overly["image"][:, :, :-1], dtype=cv2.CV_32F
        )
        background = cv2.multiply(
            1 - alpha_rgb,
            bckgPadded[yBot:yTop, xLeft:xRight][:, :, :-1],
            dtype=cv2.CV_32F,
        )
        bckgPadded[yBot:yTop, xLeft:xRight][:, :, :-1] = cv2.add(foreground, background)
        bckgPadded[yBot:yTop, xLeft:xRight][:, :, -1] = alpha
        # Keep only the original picture without the padding
        bckg = bckgPadded[
            hO : hO + bckg.shape[0], wO : wO + bckg.shape[1],
        ]
        cv2.imwrite("yalah.png", bckg[:, :, :])

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
        bckgWithOverlies["image"] = bckg
        try:
            bckgWithOverlies["bboxes"].append(
                [(xAnchor - wO) / wB, (yAnchor - hO) / hB, wO / wB, hO / hB]
            )
            bckgWithOverlies["category_id"].append(overly["category_id"][0])
        except AttributeError:
            bckgWithOverlies["bboxes"] = [
                [(xAnchor - wO) / wB, (yAnchor - hO) / hB, wO / wB, hO / hB]
            ]
            bckgWithOverlies["category_id"] = [overly["category_id"][0]]

    # # view
    # # imgaug.imshow(BoundingBoxesOnImage(targetBboxes, bckg.shape).draw_on_image(bckg, color = (0,255,0,255)))
    # # check if bounding boxes are within the picture
    # i = 0
    # for bbox in targetBboxes:
    #     originalArea = bbox.area
    #     remainingArea = bbox.clip_out_of_image(
    #         bckg.shape
    #     ).area  # this computes the remaining area when cut
    #     if remainingArea / originalArea < 0.4:
    #         targetBboxes.pop(i)
    #         # don't increment i if popping element
    #     else:
    #         targetBboxes[i] = bbox.clip_out_of_image(bckg.shape)
    #         i += 1

    return bckgWithOverlies


BOX_COLOR = (255, 0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def draw_bbox(annotations, category_id_to_name):

    img = annotations["image"].copy()
    color = BOX_COLOR
    thickness = 2

    for idx, bbox in enumerate(annotations["bboxes"]):
        class_id = annotations["category_id"][idx]
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
        class_name = category_id_to_name[class_id]
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


def makeOverly(image, id=None):
    overly = {key: None for key in ["image", "bboxes", "category_id"]}
    overly["image"] = cv2.imread(
        str(image), cv2.IMREAD_UNCHANGED
    )  # Keep the alpha (transparancy) channel

    h = overly["image"].shape[0]
    w = overly["image"].shape[1]
    bboxMargin = (
        2  # in px, 1px is minimum otherwhise albumentation complains when augmenting
        # 2px avoids rounding issues (greater than 1.0 for box size, ...)
    )
    hbbox = h - 2 * bboxMargin  # bbox takes whole image minus margin
    wbbox = w - 2 * bboxMargin
    overly["bboxes"] = [
        [0.5, 0.5, wbbox / w, hbbox / h,]
    ]  # yolo format [xcenter, ycenter, width, height] all relative to total picture size
    overly["category_id"] = [id]
    return overly


def rbga2rbg(img):
    trans_mask = img[:, :, 3] == 0
    img[trans_mask] = [255, 255, 255, 255]
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


def resizeYoloV3(img, dim):
    # resize only (to a multiple of YoloV3 stride (32))
    h, w = img.shape
    maxDim = np.argmax([h, w])
    ar = h / w
    if maxDim == 0:  # if maxDim is axis 0, resize axis 1
        newSize = dim, round(dim / ar)  # maybe not divisible by 32 ...
        quo, rest = divmod(newSize[1], 32)
        newSize = (
            dim,
            32 * quo,
        )  # therefore make it divisible by 32 even if modifying aspect ratio (ar)
    else:  # if maxDim is axis 1, resize axis 0
        newSize = round(ar * dim), dim
        quo, rest = divmod(newSize[0], 32)
        newSize = (
            32 * quo,
            dim,
        )
    return cv2.resize(img, (w, h))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay objects onto backgrounds with or without augmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--type",
        help="specify the type of overlay to be performed",
        type=str,
        choices=["overlay", "augmentedOverlay"],
        default="overlay",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--overlies",
        required=True,
        type=str,
        default=None,
        help="Path to images to be overlayed",
    )
    parser.add_argument(
        "-b",
        "--background",
        required=True,
        type=str,
        default=None,
        help="Path to images to serve as backgrounds",
    )
    parser.add_argument(
        "-r",
        "--resize",
        required=False,
        type=int,
        default=None,
        help="Resize the biggest dimension to specified size in px",
    )
    parser.add_argument(
        "--noAlpha",
        required=False,
        action="store_true",
        help="Convert the image to rgb instead of rgba",
    )
    parser.add_argument(
        "--saveBBOX",
        required=False,
        action="store_true",
        help="save the images with bbox displayed",
    )

    args = parser.parse_args()

    overliesPath = Path(args.overlies)
    backgroundPath = Path(args.background)

    overlies = []
    id_to_categry = {}

    for i, overly in enumerate(overliesPath.iterdir()):

        overlies.append(makeOverly(overly, id=i))
        id_to_categry[i] = overly.stem

        # visualize
        # visualize(overliesData, id_to_categry)

    trainingSetPath = Path("trainingSet")
    if not trainingSetPath.exists():
        trainingSetPath.mkdir()

    perBackground = 100
    for background in backgroundPath.iterdir():
        if background.is_file():
            backgroundImg = cv2.imread(str(background), cv2.IMREAD_UNCHANGED)
            if args.resize:
                backgroundImg = resizeYoloV3(backgroundImg, args.resize)
            for j in range(perBackground):
                # schedule augmentation of overlies
                rotate = albumentations.augmentations.transforms.RandomRotate90(p=0.25)
                aug = Compose(
                    [rotate],
                    bbox_params=BboxParams(
                        format="yolo",
                        min_area=0,
                        min_visibility=0.5,
                        label_fields=["category_id"],
                    ),
                )
                # perform augmentation on overlies
                overliesPerBackground = max(10, np.random.choice(20))
                overliesPerOverly = int(round(overliesPerBackground / len(overlies)))
                augmented = []
                for overly in overlies:
                    temp = [aug(**overly) for i in range(overliesPerOverly)]
                    augmented = augmented + temp

                # overlay augmented overlies on background
                bckgWithOverlies = overlay(augmented, backgroundImg)

                if args.noAlpha:
                    bckgWithOverlies["image"] = rbga2rbg(bckgWithOverlies["image"])

                cv2.imwrite(
                    str(trainingSetPath.joinpath("train_sample_" + str(j) + ".png")),
                    bckgWithOverlies["image"],
                )
                with open(
                    trainingSetPath.joinpath("train_sample_" + str(j) + ".txt"),
                    mode="wt",
                ) as f:
                    for i, bbox in enumerate(bckgWithOverlies["bboxes"]):
                        yolo = "{:d} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                            *[bckgWithOverlies["category_id"][i]] + bbox
                        )
                        f.write(yolo)
                if args.saveBBOX:
                    bckg_with_bbox = draw_bbox(bckgWithOverlies, id_to_categry)
                    savePath = Path("trainingSet_with_bbox").joinpath(
                        "train_sample_" + str(j) + ".png"
                    )
                    if not savePath.parent.exists():
                        savePath.parent.mkdir(parents=True)
                    cv2.imwrite(
                        str(savePath), bckg_with_bbox,
                    )


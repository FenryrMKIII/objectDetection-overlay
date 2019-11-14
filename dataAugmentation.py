import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import os

def overlay(sources, sourceBboxes, bckg):
    """
    overlay a source image containing bounding boxes onto a background
    
    :param arg1: the source image(s)
    :param arg2: the bounding boxe(s) around the source image(s)
    :type arg1: list of cv2 image(s) with optional alpha channel
    :type arg2: list of BoundingBox
    :return: the background with overlayed sources and associated bounding boxes
    :rtype: cv2 image, list of BoundingBox
    
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
    targetBboxes = []
    for source, sourceBbox in zip(sources, sourceBboxes):
        
        # reshape source in order to limit its size on the bckg
        # ratio is chosen based on the width i.e. x-axis (arbitrarily)
        ratioChoice = np.arange(0.2, 0.5, step = 0.01)
        ratioChosen = np.random.choice(ratioChoice)
        currentRatio = source.shape[1] / bckg.shape[1]
        factor = currentRatio/ratioChosen
        resizer = iaa.Resize({"height": int(source.shape[0]/factor), 
                              "width": "keep-aspect-ratio"}) # if not (int) I get a memory error often even with factor = 1 ... Weird
        source, sourceBbox = resizer.augment(image=source, bounding_boxes = sourceBbox)
        
        # remember the alpha channel of the source if one exists
        try:
            alpha = source[:,:,3]/255
        except IndexError:
            print("Source image has no alpha channel," 
                  "overlayed source will not feature any transparency")
                        
        # create a padded version of the background
        # allowing the source picture to sit on the edges i.e.
        # to be partially outside the picture
        # the padding equal to the source width and height i.e.
        # actually allows for the source to be located completely outside
        # the background
        heightSource = source.shape[0]
        widthSource = source.shape[1]
        bckgPadded = cv2.copyMakeBorder(bckg, heightSource, heightSource, widthSource, widthSource, cv2.BORDER_CONSTANT)
        
        # Anchor is chosen randomly within the padded image i.e. the inclusion
        # is allowed to be made outside the border of the bckg image
        # this is to allow partial bounding box
        # supposedly, this would help the training
        xAnchor = np.random.choice(np.arange(round(widthSource/2), bckgPadded.shape[1]-round(widthSource/2))) # anchor represents the center. Therefore, only allowed between these bounds in the padded image
        yAnchor = np.random.choice(np.arange(round(heightSource/2), bckgPadded.shape[0]-round(heightSource/2)))
        
        # compute the boundaries of the source image inside the total image
        # This is performed assuming that in the source image, the 
        # object center is perfectly in the middle of the source image &
        # associated bounding box
        # Therefore, one only to add/remove half the height and width with respect
        # to the chosen anchor to find the two opposite corners of the 
        # source/bounding box inside the padded background
        xLeft = int(round(xAnchor - widthSource/2))
        xRight = int(round(xAnchor + widthSource/2))
        yBot = int(round(yAnchor - heightSource/2))
        yTop = int(round(yAnchor + heightSource/2))
        
        # can happen due to roundings that the shape of the image in the 
        # computed location exceeds/is smaller than original image
        # correct that
        if ((xRight-xLeft > widthSource) | (xRight-xLeft < widthSource)) :
            xLeft = xRight - widthSource
        if ((yTop-yBot > heightSource) | (yTop-yBot < heightSource)):
            yBot = yTop - heightSource
        # can also happen that when recomputing the location,
        # the coordinates go negative which is impossible
        # therefore, make coordinate positive and then 
        # translate by an amount equal to the negative value of the
        # coordinate
        if xLeft < 0:
            xLeft = abs(xLeft)
            delta = 2*xLeft
            xRight = xRight + delta
        if yBot < 0:
            yBot = abs(yBot)
            delta = 2*yBot
            yTop = yTop +  delta
        
        # perform overlay at chosen location
        # takes into account alpha channel of the source
        if alpha.any():
            alphaMask = alpha       
                                     # augmentation such as blur, ... touches
                                     # the alpha channel values
                                     # therefore one has to be more relaxed
                                     # on the condition
                                     # if picture was perfectly sharp, alpha
                                     # channel value should be either
                                     # 255 (no transparency at this pixel)
                                     # or 0 (full transparency at this pixel)
                                     # based on tests, 50 seems to work OK
            alphaMask = np.expand_dims(alphaMask, axis=2)
        # if source has no alpha channel
        # simply create a dummy mask with shape (heightSource, widthSource)
        # that sets the alpha channel as 255 everywhere i.e. 
        # include the entire source image into the background without 
        # any transparency
        else:
            alphaMask = np.ones(source.shape[:2])
        bckgPadded[yBot:yTop, xLeft:xRight, :] = (alphaMask*source[:,:,:] +
        (1-alphaMask)*bckgPadded[yBot:yTop, xLeft:xRight, :])
        #bckgPadded[yBot:yTop, xLeft:xRight, :] = np.where(alphaMask, source, bckgPadded[yBot:yTop, xLeft:xRight,:])
        # In source image, bounding box is around the entire image
        # and assumed at the middle therefore one computes the distance between
        # the center of the bounding box and its edges
        # note the convention in imgAug : x-axis grows from left to right
        # y-axis grows from top to bottom
        xCtr = source.shape[1]/2
        yCtr = source.shape[0]/2
        deltaX = abs(sourceBbox.bounding_boxes[0].x1_int - xCtr)
        deltaY = abs(sourceBbox.bounding_boxes[0].y1_int - yCtr)
        
        
        # Now define the new bounding box associated to the overlayed source
        # inside the background image. The associated bboxes will be placed 
        # in the original (non-padded) background referential so 
        # define the bbox coordinates consistently i.e. removing
        # widthSource and heightSource from the coordinates
        # make a drawing to vizualize if needed for understanding
        targetBboxes.append(BoundingBox(x1=xAnchor-deltaX-widthSource, 
                                        x2 = xAnchor-
                                        deltaX+sourceBbox.bounding_boxes[0].width
                                        -widthSource, 
                    y1=yAnchor-deltaY-heightSource, 
                    y2 = yAnchor-deltaY+sourceBbox.bounding_boxes[0].height
                    -heightSource))
        # Keep only the original picture without the padding
        bckg = bckgPadded[heightSource:heightSource+bckg.shape[0],
                                   widthSource:widthSource+bckg.shape[1]]
        
    # check if bounding boxes are within the picture
    i = 0
    for bbox in targetBboxes:
        originalArea = bbox.area
        remainingArea = bbox.clip_out_of_image(bckg.shape).area # this computes the remaining area when cut
        if remainingArea/originalArea < 0.4:
            targetBboxes.pop(i)
            # don't increment i if popping element
        else :
            targetBboxes[i] = bbox.clip_out_of_image(bckg.shape)
            i+=1
        
    return bckg, targetBboxes

# define the symbol to be recognized by YOLO
# at the moment, code can only handle ONE symbol
symbolsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), r"rawSymbols")
symbols = []
for symbol in os.listdir(symbolsPath):
    if symbol.endswith("png") :
        symbols.append(cv2.imread(os.path.join(symbolsPath, symbol),
                    cv2.IMREAD_UNCHANGED)) # read with alpha channel

# since manages only one symbol,
# change list to single
symbol = symbols[0]

# and define its bounding box i.e. surrounding the complete picture symbol
# note that to avoid issues, the top left anchor of bounding box is 
# positioned just after first pixel i.e. (+margin, +margin) relative to (0,0)
# and bottom right anchor is positioned just before last pixel
# i.e. (-margin,-margin) relative to image width and height
margin = 5
bbox = BoundingBoxesOnImage([
    BoundingBox(x1=0+margin, y1=0+margin, 
                x2=symbol.shape[1]-margin, y2=symbol.shape[0]-margin)],  # position bounding box just before last pixel
                shape=symbol.shape)

# take care if one wants to plot the bounding box
# source has an alpha channel and imgAug don't directly plot correctly
# such picture. Manipulation is required otherwhise picture & boxs will appear
# all black. Example is shown below
#alphaMask = symbol[:,:,3] == 255
#bChannel = 255*alphaMask
#symbolBlue = symbol.copy()
#symbolBlue[:,:,2] = bChannel
#ia.imshow(symbolBlue)
#ia.imshow(symbol)
#ia.imshow(bbox.draw_on_image(symbolBlue[:,:,:3], size=2)) # view correctly
#ia.imshow(bbox.draw_on_image(symbol[:,:,:3], size=2)) # view all black


# define the desired augmentation
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    #iaa.Sometimes(1.0,
    #    iaa.GaussianBlur(sigma=(0, 10.0))
    #),
    # Strengthen or weaken the contrast in each image.
    #iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=False) # apply augmenters in random order

#seq = iaa.Sequential([
#        iaa.WithChannels([0,1,2,3],iaa.Affine(
#                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                rotate=(-25, 25),
#                shear=(-8, 8))), 
#        iaa.WithChannels([0,1,2,3], iaa.GaussianBlur(sigma=(3.0,10.0)))
#                        ], random_order=False)

# create an array of identical pictures to be augmented
# with corresponding bounding boxes
nbImg = 32
symbols = np.array([symbol for _ in range(nbImg)],dtype=np.uint8)
symbolsBbox = [bbox for _ in range(nbImg)]
#
#image_aug, bbs_aug = seq.augment(images = symbols, 
#                                         bounding_boxes=symbolsBbox)
#
#for img in image_aug:
#    alphaMask = img[:,:,3] > 50
#    bChannel = 255*alphaMask
#    imgBlue = img.copy()
#    imgBlue[:,:,2] = bChannel
#    ia.imshow(imgBlue[:,:,:])

# Initialize the loop & launch
i = 0
backgroundPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), r"backGrounds")
trainingPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'trainingSet/')
try:
    os.makedirs(trainingPath)
except FileExistsError:
    for root, dirs, files in os.walk(trainingPath, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(trainingPath)
    os.makedirs(trainingPath)
overLayedPictures = []
perBackground = 100
for background in os.listdir(backgroundPath):
    background = os.path.join(backgroundPath, background)
    for j in range(perBackground): # choose the number of times to use a background
                            # for generating a training picture
        backgroundCV2 = cv2.imread(background)
        backgroundCV2 = cv2.resize(backgroundCV2, (416, 416)) 
        # add alpha channel to background
        backgroundCV2 = np.concatenate((backgroundCV2, np.ones((backgroundCV2.shape[0], backgroundCV2.shape[1],1))*255), axis = 2)
        image_aug, bbs_aug = seq.augment(images = symbols, 
                                         bounding_boxes=symbolsBbox)
        bbs_aug = np.array([bbs.remove_out_of_image().clip_out_of_image() for bbs in bbs_aug]) # transforming to array instead of list for easier slicing
    
        # choose to include between 1 to 5 symbols in a background picture
        nbSymbols = max(1, np.random.choice(5))
        # then sample from the available augmented symbols and overlay them to the background picture
        samples = np.random.choice(np.array(np.arange(image_aug.shape[0])), nbSymbols)
        bckgWithSymbols, bboxes = overlay(image_aug[samples,:,:,:], bbs_aug[samples], backgroundCV2)
        overLayedPictures.append((bckgWithSymbols, bboxes))
        
        # write background with bboxes in YOLO format
        # png format is important to preserve alpha channel
        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), r'trainingSet/training_sample_' + str(i) + '.png'), bckgWithSymbols)
        bboxYOLO = np.zeros((len(bboxes),5))
        j = 0
        for bbox in bboxes:
            bboxYOLO[j,0] = 0 # label
            bboxYOLO[j,1] = bbox.center_x/bckgWithSymbols.shape[1] # xcenter relative
            bboxYOLO[j,2] = (bckgWithSymbols.shape[0] - bbox.center_y)/bckgWithSymbols.shape[0] # ycenter relative !yolo convention for y axis is growing from the bottom i.e. opposite to imgAug so adapt
            bboxYOLO[j,3] = bbox.width/bckgWithSymbols.shape[1] # bbox width relative
            bboxYOLO[j,4] = bbox.height/bckgWithSymbols.shape[0] # bbox height relative
            j+=1
        np.savetxt(fname = os.path.join(trainingPath, 'training_sample_' + 
                                        str(i) + '.txt'), X = bboxYOLO,
                    fmt=['%d','%.2f','%.2f','%.2f','%.2f'])
        i+=1


## check some pictures
#toCheck = [0,5,10]
#for item in toCheck:
#    bboxes = overLayedPictures[item][1]
#    picture = overLayedPictures[item][0]
#    ia.imshow(picture) # throwing a range error but casting astype(np.uint8) solves the issue.
#                        # should look at the code where to this properly
#    #ia.imshow(BoundingBoxesOnImage(bboxes, shape=picture.shape).draw_on_image(picture, size=2))


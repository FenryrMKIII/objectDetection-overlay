import cv2
import importlib
import imgaug
importlib.reload(imgaug)
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import os
import brambox
from PIL import Image

def overlay(sources, sourceBboxes, bckg):
    """
    overlay a source image containing bounding boxes onto a background
    
    Parameters:
    -----
    sources(list of numpy arrays): the source image(s), potentially augmented (rotated, scaled, ...)
    sourceBboxes(list of imgaug.augmentables.bbs.BoundingBoxesOnImage) : list of the bounding boxe(s) associated to the source image(s)
    bckg (list of numpy arrays) : the background on which the source images will be overlayed

    Returns:
    --------
    bckg (numpy array) : the background with overlayed sources images
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
    targetBboxes = []
    for source, sourceBbox in zip(sources, sourceBboxes):
        # view
        #imgaug.imshow(sourceBbox.draw_on_image(source, color=(0, 255, 0, 255)))

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
            if bckg.shape[-1] != 4:
                # Have to use PIL because ... Don't know why
                # Performing the operation using numpy arrays 
                # i.e. np.concatenate((bckg,np.expand_dims(np.ones(bckg.shape[:-1])*255,axis=2)), axis=2)
                # somehow does not work (image appears fully white)
                # altough the produced numpy array is identical to what PIL produces and saving it to a png
                # e.g. matplotlib.image.imsave('name.png', myArray/255) produces the
                # desired picture
                bckg = np.array(Image.fromarray(bckg).convert('RGBA'))
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
        
        # Anchor is chosen randomly within the padded image 
        # but without allowing the picture to lie outside the original image
        # in other words, the padding is not used at present
        # padding shall be used in a further step to allow partial image to be 
        # detected (this supposedly would help training) but this requires
        # to recompute bounding boxes after inclusion (otherwhise, center of
        # bbox could be outside the picture !) 
        easy = False
        if easy :
            xAnchor = round(bckgPadded.shape[1]/2)
            yAnchor = round(bckgPadded.shape[0]/2)
        else :
            xAnchor = np.random.choice(np.arange(round(1.5*widthSource), bckgPadded.shape[1]-round(1.5*widthSource))) # anchor represents the center. Therefore, only allowed between these bounds in the padded image
            yAnchor = np.random.choice(np.arange(round(1.5*heightSource), bckgPadded.shape[0]-round(1.5*heightSource)))
        # note the round hereabove are for future when widthSource/2 will have to be used
        # to allow partial inclusion of bbox
        
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
        try:
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
        except NameError:
            alphaMask = np.ones(source.shape[:2])
            alphaMask = np.expand_dims(alphaMask, axis=3)

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
    # view
    #imgaug.imshow(BoundingBoxesOnImage(targetBboxes, bckg.shape).draw_on_image(bckg, color = (0,255,0,255)))
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

# Then define bounding box (manual definition)
bboxes = [BoundingBoxesOnImage([
        BoundingBox(x1=860, y1=860, 
                x2=1720, y2=1533)],  # position bounding box just before last pixel
                shape=symbols[0].shape),
        BoundingBoxesOnImage([
        BoundingBox(x1=819, y1=769, 
                x2=1831, y2=1499)],  # position bounding box just before last pixel
                shape=symbols[1].shape),
        BoundingBoxesOnImage([
        BoundingBox(x1=781, y1=565, 
                x2=1767, y2=1675)],  # position bounding box just before last pixel
                shape=symbols[2].shape)]

# for symbol, bbox in zip(symbols, bboxes):
#     imgaug.imshow(bbox.draw_on_image(symbol, color=(0, 255, 0, 255)))

# and define its bounding box i.e. surrounding the complete picture symbol
# note that to avoid issues, the top left anchor of bounding box is 
# positioned just after first pixel i.e. (+margin, +margin) relative to (0,0)
# and bottom right anchor is positioned just before last pixel
# i.e. (-margin,-margin) relative to image width and height


# Visualize bbox

# take care if one wants to plot the bounding box
# source has an alpha channel and imgAug don't directly plot correctly
# such picture. Manipulation is required otherwhise picture & boxs will appear
# all black. Example is shown below
# alphaMask = symbol[:,:,3] == 255
# bChannel = 255*alphaMask
# symbolBlue = symbol.copy()
# symbolBlue[:,:,2] = bChannel



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
        rotate=(-180, +180),
        shear=(-8, 8)
    )
    #,iaa.Pad(100)
], random_order=False) # apply augmenters in random order

#seq = iaa.Sequential([
#        iaa.WithChannels([0,1,2,3],iaa.Affine(
#                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                rotate=(-25, 25),
#                shear=(-8, 8))), 
#        iaa.WithChannels([0,1,2,3], iaa.GaussianBlur(sigma=(3.0,10.0)))
#                        ], random_order=False)

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
        if os.path.isfile(background):
            backgroundCV2 = cv2.imread(background)
            backgroundCV2 = cv2.resize(backgroundCV2, (416, 416)) 

            # choose to include between 0 to 3 instances from each symbol in a background picture
            symbolsForAug = []
            symbolsBboxForAug = []
            for symbol, bbox in zip(symbols, bboxes):
                nbInstance = max(0, np.random.choice(3))
                symbolsForAug += [symbol for _ in range(nbInstance)]
                symbolsBboxForAug += [bbox for _ in range(nbInstance)]
            symbolsForAug = np.array(symbolsForAug,dtype=np.uint8)
            if symbolsForAug.shape[0] != 0:
                image_aug, bbs_aug = seq.augment(images=symbolsForAug, bounding_boxes=symbolsBboxForAug)
                bbs_aug = np.array([bbs.remove_out_of_image().clip_out_of_image() for bbs in bbs_aug]) # transforming to array instead of list for easier slicing
                bckgWithSymbols, bboxesOnBackground = overlay(image_aug, bbs_aug, backgroundCV2)
            else:
                # no symbols overlayed
                bckgWithSymbols = backgroundCV2
                bboxesOnBackground = []
            overLayedPictures.append((bckgWithSymbols, bboxesOnBackground))
            
            # write background with bboxes in YOLO format
            # png format is important to preserve alpha channel
            cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), r'trainingSet/training_sample_' + str(i) + '.png'), bckgWithSymbols[:,:,:])
            bboxYOLO = np.zeros((len(bboxesOnBackground),5))
            j = 0
            for bbox in bboxesOnBackground:
                bboxYOLO[j,0] = 0 # label
                bboxYOLO[j,1] = bbox.center_x/bckgWithSymbols.shape[1] # xcenter relative
                bboxYOLO[j,2] = bbox.center_y/bckgWithSymbols.shape[0] #(bckgWithSymbols.shape[0] - bbox.center_y)/bckgWithSymbols.shape[0] # ycenter relative !yolo convention for y axis is growing from the bottom i.e. opposite to imgAug so adapt
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


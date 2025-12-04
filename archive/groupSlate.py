import cv2
import os
import pupil_apriltags
import PIL
import transformers
import numpy as np
from PIL import Image
from pupil_apriltags import Detector
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#print("hello world")
processer = None
model = None

'''extractSlateImg(filename, debug = False)
filename: name of file in current directory
creates an image of just the slate that has been made straite on'''
def extractSlateImg(filename, debug = False):
    image = cv2.imread(filename)
    #Convert the image to grayscale 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    print("extracting slate image")
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    #what is this, will it help??
    #arucoParams = cv2.aruco.DetectorParameters()
    #arucoParams.markerBorderBits = 2
    #params.adaptiveThreshWinSizeStep = 1

    d = at_detector.detect(image)
    print(f"len d: {len(d)}")
    if len(d) == 0:
        if debug:
            print("no tag")
        return None
    
    aprCent = list(d[0].center)
    #print(aprCent)
    #cv2.circle(image, (int(aprCent[0]), int(aprCent[1])), 50, (255, 0, 0), 5) 
    #draws circle over april tag, when the slate is small in frame this can cover the writing causing bugs
    #cv2_imshow(image)

    homoGr = d[0].homography #list?
    #print("homoGr")
    #print(homoGr)

    homography, status = cv2.findHomography(d[0].corners, np.array([[0, 0], [100, 0], [100, 100], [0, 100]])) #[0, 0], [100, 0], [100, 100], [0, 100]
    #print("status")
    #print(status)

    #print("homography") # mult x y 1
    #print(homography)
    warp_img = cv2.warpPerspective(image, homography, (470, 360)) #demensions of the slate transformed into
    #im_dest = cv2.warpPerspective(image, homoGr, (100000, 100000))
    #resized_image_dest = cv2.resize(im_dest, (100, 100)) 

    #depending on april tag orientation flip it to right side up

    '''this is entirly about how the april tag is placed 
        we can prob remove once no more april tags'''
    #flipWarp_img = cv2.flip(warp_img, 0) #use this line to flip over y axis
    flipWarp_img = cv2.flip(warp_img, 0) #use to flip over x axis

    return flipWarp_img

'''TBH I don't remember what this does
finds what the thing (portion of image) says'''
def ocr_image(src_img):
    global processer
    global model
    if not processer:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
    if not model:
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  
'''gets the actual text from the image chunck'''
def ocrTextInRectangle(img, rect, debug = False, showImg = False):
    #rect is: [0, 100, 50, 200] for [x start, x len, y start, y len]
    y,height,x,width = rect
    crop_img = img[x:x+width, y:y+height]
    plt.imshow(crop_img, cmap='gray', vmin=0, vmax=255)
    print("saving crop img")
    cv2.imwrite('text_{}.jpg'.format(y), crop_img)
    result = cv2.imwrite('./test.png', crop_img) #cv2.imwrite('/Downloads/SlateT1/image1.png', image1)
    #print(f'tried to save image {result}: {crop_img}')
    print(f'saved image {result}')
    #if showImg:
        #cv2.imshow("Cropped Image", crop_img) #cv2.imshow(crop_img)
    img = Image.fromarray(crop_img)
    color_img = img.convert('RGB')
    #print("saving crop")
    #cv2.imwrite('slateCrop_{}.jpg'.format(f"recognized-as-{color_img}"), crop_img)
    #cv2.imwrite('/Downloads/SlateT1/testImg.png', color_img)
    r = ocr_image(color_img)
    #print("saving r")
    #cv2.imwrite('/Downloads/SlateT1/testImg.png', r)
    if debug:
        print("debug")
        cv2.imwrite('slateCrop_{}.jpg'.format(f"recognized-as-{r}"), crop_img)
    return r

'''given where the scene and take should be written relative to prob the top left
find the word thats in that box given a box size'''
def extractSceneAndTake(exractedSlate, debug = False):
    #rect is: [0, 100, 50, 200] for [x start, x len, y start, y len]
    #These numbers determine the box the take and scene will be read from in the image
    #scene = ocrTextInRectangle(exractedSlate, ***[140, 150, 260, 85]*** [140, 130, 100, 130], debug)
    #take = ocrTextInRectangle(exractedSlate, [360, 100, 260, 85] [280, 170, 110, 120], debug)
    '''all the nums in the lists should probably be variables but they will need to be manual I assume 
    which is the reason they are currently hard coded
    these values should probably calculated via proportion to account for diff images haveing diff amount of pixles'''
    scene = ocrTextInRectangle(exractedSlate, [850, 1000, 550, 600], debug) #showImg only for colab
    take = ocrTextInRectangle(exractedSlate, [2000, 500, 550, 600], debug) #600, 100, 200, 100
    return (scene, take)

'''get frames of the video file(s) given. This has a lot of room for improvment in terms of algorithm
it could be alowed to look forever or stop after a set time
the current solution I have is: check every 10 frames till you find the tag (we would change the find condition)
keep extracting scene and take. 
only stop if you reach the end
or
there is a frame without a tag and you have at least 6 extractions
or
there has been at least 1 extraction and we are over 200 frames in
or
nothing has been found and we are over 500 frames in

I have lots of Ideas to add here including potential motion tracking, but mostly
looking for the slate better'''
def extractVidFrames(filename, start_frame=0, end_frame=-1, skip=10, debug = False):
    cap = cv2.VideoCapture(filename)

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(f"fps: {fps}!")
    # Calculate the interval between frames
    #interval = 1 / fps

    # Create a counter to keep track of the frame number
    frame_count = 0
    #keeps track of most recent frame with slate identified
    captFrame = 0
    #print(cap.isOpened())
    cap.isOpened() == True
    results = []
    # Loop through the frames
    while cap.isOpened():

        # Capture the next frame
        ret, frame = cap.read()

        # If the frame is empty, break out of the loop
        if not ret:
            break

        # Increment the frame counter
        frame_count += 1

        # Save the frame as an image
        if frame_count >= start_frame and frame_count%skip == 0 and ((frame_count <= end_frame) or (end_frame == -1)):
            if debug:
                print(f"frame num: {frame_count}!")
                #print(f"last captured frame num: {captFrame}!")
                cv2.imwrite('frame_{}.jpg'.format(frame_count), frame)
                frameFile = ('frame_{}.jpg'.format(frame_count))
            esi = extractSlateImg(frameFile)#matrix representing the image itself an ndarray
            #print(esi)
            #print(type(esi))
            if isinstance(esi, np.ndarray):
                st = extractSceneAndTake(esi, debug)
                captFrame = frame_count
                if debug:
                    cv2.imwrite('slate_{}.jpg'.format(frame_count), esi)
                    print(f"Found scene/take: {st}")
                results.append(st)
            elif len(results) > 6:
                if debug:
                    print("found at least 6")
                    print(f"results: {results}")
                cap.release()
                return results
            #print(f"frame_count - captFrame = {frame_count - captFrame}")
            #print(f"length of results: {len(results)}")
            if (len(results) >= 1) and ((frame_count - captFrame) > 200):
                if debug:
                    print("found at least 1, but now the slate is gone")
                    print(f"results: {results}")
                cap.release()
                print("cap release")
                return results
            if (len(results) == 0) and ((frame_count - captFrame) > 500):
                if debug:
                    print("found none, can't find slate")
                    print(f"results: {results}")
                cap.release()
                print("cap release")
                return results

            #elif esi == 0:

    #make it stop watching at a threshold
    # Release the video capture object
    if debug:
        print("went through entire video")
        print(f"results: {results}")
    cap.release()
    return results

'''clean the text'''
def cleanScene(s, debug = False):
    match = re.search(r"\d+[A-Za-z]", s) #\d+[A-Z] #
    #catches error when theres no capital Letter w/ num
    if not match:
        if debug:
            print("scene is not parsable returning none")
        return None
    return match[0]

'''clean the text'''
def cleanTake(s, debug = False):
    match = re.search(r"\d+", s)
    #catches error when theres no num
    if not match:
        if debug:
            print("take is not parsable returning none")
        return None
    return match[0]
    #TODO: make this return list of all digets separated by spaces

'''calculate the predicted value of the scene based on the mode of what was read'''
def findScene(results):
    cleanS = [cleanScene(x[0]) for x in results]
    #finds most common one (use Count dict) FUTURE: maybe combine w/ findTake
    sceneCountDict = Counter(cleanS)
    modeScene = sceneCountDict.most_common(1)
    #print(cleanS)
    #print(modeScene)
    sceneRes = modeScene[0][0]
    return(sceneRes)

'''calculate the predicted value of the take based on the mode of what was read'''
def findTake(results):
    cleanT = [cleanTake(x[1]) for x in results]
    #finds most common one
    takeCountDict = Counter(cleanT)
    modeTake = takeCountDict.most_common(1)
    #TODO: edge case when mode is a tie and or low mode
    #print(cleanT)
    #print(modeTake)
    takeRes = modeTake[0][0]
    return(takeRes)

'''give the video its new name after finding what it should be labeled'''
def proccessAndRenameVid(filename, debug = False):
    scene,take = proccessVideo(filename, debug)
    #TODO: rename here
    dir, f = os.path.split(filename)
    newName = os.path.join(dir, f"{scene}.{take}.{f}")
    print(f"renaming {filename} to {newName}")
    os.rename(filename, newName)

'''edge cases and doing the thing'''
def proccessVideo(filename, debug = False):
    results = extractVidFrames(filename, debug=debug)
    if len(results) <= 2:
        s = "-scene-"
        t = "-take-"
    else:
        s = findScene(results)
        t = findTake(results)
    if debug:
        print(f"----processing results for {filename}----")
        print(results)
        print()
        print(s)
        print(t)
    return s, t
    
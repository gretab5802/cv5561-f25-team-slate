'''
What should be occuring in this file

- take in video file
- extract frames
- find slate in first frame using sift
- align slate to template
- extract text from slate using trocr
- move on to next frame
- do this for all frames that were extracted
- compile generated texts into array
- take the name that came up the most
- rename video file based on that extracted text

authors:
- Anthony Camano-Enriquez
-
-
'''

import numpy as np
import cv2 as opencv
from PIL import Image
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import helperfunctions as helpers

if __name__=='__main__':
    ## getting template image
    print('loading template image...\n');
    template = helpers.getTemplateImage();

    ## extract frames from video
    videoFilePath = 'test_video.mp4'; ## change this for different video files
    frameInterval = 6; ## extract every 6th frame

    print(f'extracting frames from {videoFilePath}...\n');
    extractedFrames = helpers.extractFrames(videoFilePath, frameInterval);


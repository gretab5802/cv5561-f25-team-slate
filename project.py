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
- Anthony
- Genevieve
- Greta
'''

import os
from collections import Counter

import cv2 as opencv
import matplotlib.pyplot as plt

import helperFunctions as helpers

# ----- Global Parametsr -----

videoFilePath = 'videos/8A.mp4'; ## change this for different video files
frameInterval = 6; ## extract every 6th frame (must be multiple of 2)
breakProcessingEarly = True; ## set to True to only process first 10 frames
showVisualizations = False; ## set to True to display viz
# trOCRModelName = 'microsoft/trocr-base-handwritten'; ## model for text extraction
trOCRModelName = 'microsoft/trocr-large-handwritten'; ## model for text extraction

# ----- Main Project Function -----

if __name__=='__main__':
    ## getting template image
    print('loading template image...\n');
    template = helpers.getTemplateImage();

    ## extract frames from video
    print(f'extracting frames from {videoFilePath}...\n');
    extractedFrames = helpers.extractFrames(videoFilePath, frameInterval);
    
    print(f'extracted {len(extractedFrames)} frames\n');
    
    ## convert extracted frames to grayscale for processing
    print('converting frames to grayscale...\n');
    grayscale_frames = [];
    for frame in extractedFrames:
        gray_frame = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY);
        grayscale_frames.append(gray_frame);
    
    ## track template across all frames
    print('tracking slate across frames...\n');
    A_list, errors_list = helpers.trackMultiFrames(template, grayscale_frames, breakProcessingEarly);
    
    print(f'processed {len(A_list)} frames\n');
    
    ## warp each frame to align the slate with the template
    print('warping frames to align slates...\n');
    warped_frames = [];
    for i, (gray_frame, A) in enumerate(zip(grayscale_frames, A_list)):
        warped_frame = helpers.warpImage(gray_frame, A, template.shape);
        warped_frames.append(warped_frame);
        print(f'  warped frame {i+1}/{len(A_list)}');
    
    print(f'\nwarped {len(warped_frames)} frames\n');

    ## display template and all warped frames if visualization is enabled
    if showVisualizations:
        print('displaying template and all warped frames...\n');
    
        num_frames = len(warped_frames);
        cols = min(4, num_frames + 1);  ## Max 4 columns
        rows = (num_frames + 1 + cols - 1) // cols;  ## Calculate rows needed
    
        plt.figure(figsize=(cols * 4, rows * 3));
    
        ## Show template in first subplot
        plt.subplot(rows, cols, 1);
        plt.imshow(template, cmap='gray', vmin=0, vmax=255);
        plt.title('Template');
        plt.axis('off');
    
        ## Show all warped frames
        for i, warped_frame in enumerate(warped_frames):
            plt.subplot(rows, cols, i + 2);
            plt.imshow(warped_frame, cmap='gray', vmin=0, vmax=255);
            plt.title(f'Warped Frame {i+1}');
            plt.axis('off');
    
        plt.tight_layout();
        plt.show();
    
    ## extract text from aligned slates using TrOCR
    print('extracting text from warped frames...\n');
    extracted_texts = helpers.extractTextFromFrames(warped_frames, trOCRModelName, showVisualizations);
    
    print(f'\nExtracted + Processed texts: {extracted_texts}\n');
    
    ## find most common extracted text
    if len(extracted_texts) == 0:
        print("No text extracted, leaving original video file unchanged.")
    else:
        # Counter just counts most frequent item in a list but is nice because
        # you don't have to write a loop and has useful built in functions
        counts = Counter(extracted_texts)
        most_common_text, most_common_count = counts.most_common(1)[0]
        print(f"Most common extracted text: {most_common_text}")

        # build new filename
        original_dir = os.path.dirname(videoFilePath)
        original_basename = os.path.basename(videoFilePath)      # this gets us "8A.mp4"
        original_stem, original_ext = os.path.splitext(original_basename)  # gives us "8A", ".mp4"

        new_basename = f"{most_common_text}.{original_stem}{original_ext}"  # append most common text found
        new_path = os.path.join(original_dir, new_basename)

        # rename file
        try:
            os.rename(videoFilePath, new_path)
            print(f"Renamed video file to: {new_path}")
        except OSError as e:
            print(f"Failed to rename file: {e}")


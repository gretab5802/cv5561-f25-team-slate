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

import cv2 as opencv
import matplotlib.pyplot as plt

import helperfunctions as helpers

# ----- Global Parametsr -----

videoFilePath = 'videos/testing.MP4'; ## change this for different video files
filename = 'testing.MP4' ## change this for different video files, must match above
frameInterval = 6; ## extract every 6th frame (must be multiple of 2)
breakProcessingEarly = True; ## set to True to only process first 10 frames
showVisualizations = True; ## set to True to display viz
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

    if showVisualizations:
        print('displaying template and all extrated frames...\n');
    
        num_frames = len(extractedFrames);
        cols = min(4, num_frames + 1);  ## Max 4 columns
        rows = (num_frames + 1 + cols - 1) // cols;  ## Calculate rows needed
    
        plt.figure(figsize=(cols * 4, rows * 3));
    
        ## Show template in first subplot
        plt.subplot(rows, cols, 1);
        plt.imshow(template, cmap='gray', vmin=0, vmax=255);
        plt.title('Template');
        plt.axis('off');
    
        ## Show all warped frames
        for i, warped_frame in enumerate(extractedFrames):
            #?
            plt.subplot(rows, cols, i + 2);
            plt.imshow(warped_frame, cmap='gray', vmin=0, vmax=255);
            plt.title(f'Frame {i+1}');
            plt.axis('off');
    
        plt.tight_layout();
        plt.show();
    
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
        if A is not 0:
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
    extracted_scenes = helpers.extractSceneTextFromFrames(warped_frames, trOCRModelName, showVisualizations);
    extracted_takes = helpers.extractTakeTextFromFrames(warped_frames, trOCRModelName, showVisualizations);
    
    print(f'\nExtracted + Processed scene: {extracted_scenes}\n');
    print(f'\nExtracted + Processed take: {extracted_takes}\n');

    scene_result = helpers.findScene(extracted_scenes)
    take_result = helpers.findTake(extracted_takes)

    print(f'\nscene extracted is: {scene_result}\n');
    print(f'\ntake extracted is: {take_result}\n');

    helpers.renameVid("videos", filename, scene_result, take_result)

'''
helper functions included in this file:

'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from cv2 import SIFT_create, KeyPoint_convert, filter2D
from cv2 import resize
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import torch
from collections import Counter
import re
import os

def getTemplateImage():
    template = Image.open('template.jpeg');
    template = np.array(template.convert('L'));
    return template;

def getVideoFile(filepath):
    videoFile = filepath;
    return videoFile;

def extractFrames(videoFilepath, frameInterval):
    '''
    Extract frames from a video at specified intervals.
    
    Parameters:
    - videoFilepath (str): Path to the video file
    - frameInterval (int): Extract every Nth frame (e.g., 6 means extract every 6th frame)
    
    Returns:
    - extracted_frames (list): List of extracted frames as numpy arrays
    '''
    
    extractedFrames = []
    
    # Open the video file
    cap = cv2.VideoCapture(videoFilepath)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {videoFilepath}\n")
        return extractedFrames
    
    frameCount = 0
    
    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        # If frame reading was not successful, break
        if not ret:
            break
        
        # Extract frame if it's at the specified interval
        if frameCount % frameInterval == 0:
            extractedFrames.append(frame) #frame
        
        frameCount += 1
    
    # Release the video capture object
    cap.release()
    
    return extractedFrames

def findMatch(img1, img2):
    x1, x2 = None, None
    disThr = 0.7
    
    ## create sift object
    sift = SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    kp1 = KeyPoint_convert(kp1)
    kp2 = KeyPoint_convert(kp2)

    ## forward matching: img1 -> img2
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    distances, indices = neighbors.kneighbors(des1)

    ## backward matching: img2 -> img1 (for cross-check)
    neighborsBack = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    distancesBack, indices_back = neighborsBack.kneighbors(des2)

    goodMatches = []

    for i in range(len(distances)):
        disClosest = distances[i][0]
        disSecond = distances[i][1]
        
        if disSecond > 0 and (disClosest / disSecond) < disThr:
            j = indices[i][0]  ## best match in img2

            ## apply Lowe's ratio test for backward match as well
            disBackClosest = distancesBack[j][0]
            disBackSecond = distancesBack[j][1]
            
            ## cross-check: verify that img2[j]'s best match is img1[i]
            ## AND that backward match also passes ratio test
            if (indices_back[j][0] == i and 
                disBackSecond > 0 and 
                (disBackClosest / disBackSecond) < disThr):
                goodMatches.append((i, j))
    
    x1 = np.zeros((len(goodMatches), 2))
    x2 = np.zeros((len(goodMatches), 2))

    for idx, (i, j) in enumerate(goodMatches):
        x1[idx] = kp1[i]
        x2[idx] = kp2[j]

    if len(x1) < 5:
        x1 = None
        x2 = None

    return x1, x2

def alignImageUsingFeature(x1, x2, ransacThr, ransactIteration):
    A = None

    n = x1.shape[0]
    
    ## need at least 3 points for affine transform
    if n < 3:
        return np.eye(3)
    
    bestInliers = []
    bestA = None
    
    ## RANSAC iterations
    for iteration in range(ransactIteration):
        ## randomly sample 3 points
        ## np.random.choice returns indices!!!
        indices = np.random.choice(n, size=3, replace=False)
        sampleX1 = x1[indices]
        sampleX2 = x2[indices]
        
        ## compute affine transform from these 3 points
        A = computeAffineTransform(sampleX1, sampleX2)
        
        if A is None:
            continue
        
        ## transform all x1 points using this affine matrix
        ## convert x1 to homogeneous coordinates (n x 3)
        x1H = np.hstack([x1, np.ones((n, 1))])
        
        ## transform: x2Pred = A * x1
        ## x2Pred will be (n x 3), we take first 2 columns
        x2Pred = (A @ x1H.T).T
        x2Pred = x2Pred[:, :2]  ## remove homogeneous coordinate
        
        ## compute errors (Euclidean distance)
        ## errors[i] = distance between predicted and actual point
        errors = np.sqrt(np.sum((x2Pred - x2)**2, axis=1))
        
        ## find inliers (points with error < threshold)
        inliers = np.where(errors < ransacThr)[0]
        
        ## keep track of best model (one with most inliers)
        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            bestA = A
    
    ## refine the affine transform using ALL inliers
    ## this gives better accuracy than using just 3 random points
    if len(bestInliers) >= 3:
        bestA = computeAffineTransform(x1[bestInliers], x2[bestInliers])
    
    ## ff RANSAC failed to find any good model, just return identity - reminder to me that this means error and need to check somethign 
    if bestA is None:
        bestA = np.eye(3)

    return bestA

def warpImage(img, A, output_size):
    img_warped = None

    h_out, w_out = output_size
    h_in, w_in = img.shape
    
    ## create grid of (x, y) coordinates in output image 
    x_out, y_out = np.meshgrid(np.arange(w_out), np.arange(h_out))

    coords_out = np.vstack([
        x_out.ravel(),
        y_out.ravel(),
        np.ones((h_out * w_out))
    ])  # 3 x (h_out*w_out) <- what shape of coords_out should be

    ## compute inverse mapping
    A_inv = np.linalg.inv(A)

    ## ACTUALLY it seems that A is already the inverse so we dont have to invert it -.-
    coords_in = A @ coords_out  # 3 x N
    
    ## extract x and y coordinates (no division needed for affine? i think need to double chekc)
    x_in = coords_in[0, :]  # x coordinates in input image
    y_in = coords_in[1, :]  # y coordinates in input image
    
    ## define the grid for interpn (row, column)
    points = (np.arange(h_in), np.arange(w_in))
    
    ## stack coordinates as (N, 2) in (row, col) order = (y, x) order
    xi = np.column_stack([y_in, x_in])
    
    ## interpolate pixel values using interpn - this might be giving me error idk
    img_warped_flat = interpolate.interpn(
        points=points,
        values=img,
        xi=xi,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    ## reshape to output dimensions
    img_warped = img_warped_flat.reshape((h_out, w_out))

    return img_warped

def alignImage(template, target, A):
    '''  
    Input: 
        template - grayscale template image
        target - grayscale target image
        A - 3x3 initial affine transform (x_tgt = A * x_tpl)
    
    Output:
        A_refined - refined 3x3 affine transform
        errors - list of alignment errors (L2 norm) at each iteration
    '''
    A_refined = None
    errors = None

    ## parameters for ICIA
    max_iters = 100
    epsilon = 0.001
    
    # take p parameters from A -- se below for what A should look like, this was from notes
    # A = | p1+1  p2    p3 |
    #     | p4    p5+1  p6 |
    #     | 0     0     1  |
    p = np.array([A[0, 0] - 1, A[0, 1], A[0, 2],
                  A[1, 0], A[1, 1] - 1, A[1, 2]])
    
    ## use sobel filters for x and y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    
    grad_x = filter2D(template.astype(np.float32), -1, sobel_x)
    grad_y = filter2D(template.astype(np.float32), -1, sobel_y)

    ## get Jacobian dW/dp at (x; 0)
    h, w = template.shape
    
    ## create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    
    ## number of pixels
    n_pixels = h * w
    
    ## jacobian for all pixels: shape (n_pixels, 2, 6)
    ## for each pixel, we have a 2x6 matrix
    jacobian = np.zeros((n_pixels, 2, 6))
    jacobian[:, 0, 0] = u  # ∂x/∂p1
    jacobian[:, 0, 1] = v  # ∂x/∂p2
    jacobian[:, 0, 2] = 1  # ∂x/∂p3
    jacobian[:, 1, 3] = u  # ∂y/∂p4
    jacobian[:, 1, 4] = v  # ∂y/∂p5
    jacobian[:, 1, 5] = 1  # ∂y/∂p6
    
    ## compute steepest descent images
    ## flatten gradients
    grad_x_flat = grad_x.flatten()
    grad_y_flat = grad_y.flatten()
    
    ## for each pixel, compute: [grad_x, grad_y] * jacobian (2x6 matrix)
    ## result: (n_pixels, 6) matrix
    steepest_descent = np.zeros((n_pixels, 6))
    for i in range(n_pixels):
        grad_i = np.array([grad_x_flat[i], grad_y_flat[i]])
        steepest_descent[i, :] = grad_i @ jacobian[i, :, :]
    
    ## cpmute hessiam
    H = steepest_descent.T @ steepest_descent ## check dimenson of this, might be weher we get error
    
    ## iterative part of the function
    errors = []
    
    for iteration in range(max_iters):
        ## convert current parameters to affine matrix
        A_current = paramsToAffine(p)
        
        img_warped = warpImage(target, A_current, template.shape)
        
        I_err = img_warped - template
        
        ## error
        error_norm = np.sqrt(np.mean(I_err ** 2)) / 255.0  ## normalize by 255
        errors.append(error_norm)
        
        I_err_flat = I_err.flatten()
        F = steepest_descent.T @ I_err_flat
        
        delta_p = np.linalg.solve(H, F)  ## More stable than inv(H) @ F
        
        A_delta = paramsToAffine(delta_p)
        
        ## invert A_delta
        A_delta_inv = np.linalg.inv(A_delta)
        
        A_new = A_current @ A_delta_inv
        
        ## extract parameters from new affine matrix
        p = np.array([A_new[0, 0] - 1, A_new[0, 1], A_new[0, 2],
                      A_new[1, 0], A_new[1, 1] - 1, A_new[1, 2]])
        
        # check with what we set epsilon as === 0.0001
        if np.linalg.norm(delta_p) < epsilon:
            break
    
    A_refined = paramsToAffine(p)
    errors = np.array(errors)
    
    return A_refined, errors

def trackMultiFrames(template, img_list, breakProcessingEarly):
    '''
    Track template across multiple frames using inverse compositional alignment
    
    Input:
        template - grayscale template image (from first frame)
        img_list - list of consecutive grayscale images [img0, img1, img2, img3]
    
    Output:
        A_list - list of affine transforms from template to each frame
        errors_list - list of error arrays (one per frame) across iterations
    '''

    A_list = None
    errors_list = None

    ## parameters for RANSAC
    ransacThr = 3.0
    ransactIteration = 1000
    
    A_list = []
    errors_list = []
    
    original_template = template.copy()
    
    ## current template (will be updated after each frame)
    current_template = template.copy()
    
    ## initial Affine matrix
    A_prev = None
    v = 0
    
    for i, target_img in enumerate(img_list):
        
        if breakProcessingEarly and i >= 15:
            print("Breaking at frame 10 for testing purposes.")
            break

        print(f"Processing frame {i+1}/{len(img_list)}...")
        h, w = target_img.shape
        
        if i < 20:
            ## FIRST FRAME: Use feature matching + RANSAC
            #print("  Using feature matching for initialization...")
            
            x1, x2 = findMatch(current_template, target_img)
            if x1 is None:
                print('bad sift')
                continue
            
            A_init = alignImageUsingFeature(x1, x2, ransacThr, ransactIteration)
            #visualize_align_image_using_feature(current_template, target_img, x1, x2, A_init, ransacThr)
            # check if A is good, if it is make it the template
            cliped, squashed = is_transform_out_of_bounds(A_init, w, h)
            if cliped:
                print('transform goes out of image, not using...')
                A_list.append(0)
                continue
            if squashed:
                print('transform squashes image, not using...')
                A_list.append(0)
                continue
            valid, angle = get_affine_angle(A_init)
            ## add something about the size of the final box or like the ratio of the side lengths to each other
        
        ## refine with ICIA
        if angle > 9:
            print(f'{i} is useless angle over 9 deg')
            A_list.append(0)
            continue
        print('good, moving on')
        #A_refined, errors = alignImage(current_template, target_img, A_init)
        #visualize_align_image(current_template, target_img, A_init, A_refined)
        
        #A_list.append(A_refined)
        A_list.append(A_init)
        #errors_list.append(errors)
        
        ## save current affine for next frame's initialization
        #A_prev = A_refined.copy()
        if valid and v == 0:
            v = 1
            current_template = warpImage(target_img, A_init, current_template.shape)
            current_template = np.clip(current_template, 0, 255).astype(np.uint8)
            print('updating template')

    return A_list, errors_list

def is_transform_out_of_bounds(matrix, width, height): # or squashed
    """
    Checks if an affine transform moves any part of the image outside the canvas.
    
    Args:
        matrix: 2x3 affine transform matrix (numpy array)
        width: Image width
        height: Image height
        
    Returns:
        True if the image is clipped/out of bounds
        False if the image fits entirely inside
    """
    is_squashed = False
    m = np.array(matrix)
    
    #4 corners of the original image (x, y)
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    #turns [[x, y]] into [[x, y, 1]]
    corners_aug = np.hstack([corners, np.ones((4, 1))])
    
    transformed_corners = m.dot(corners_aug.T).T  # Result is 4x2 [[new_x, new_y], ...]
    # Check x coordinates
    min_x = transformed_corners[:, 0].min()
    max_x = transformed_corners[:, 0].max()
    
    # Check y coordinates
    min_y = transformed_corners[:, 1].min()
    max_y = transformed_corners[:, 1].max()
    side_ratio = ((max_x - min_x) / (max_y - min_y)) 
    if (side_ratio >= 1.9 )or (side_ratio <= .5):
        is_squashed = True
        print('squashed :(')
    
    is_clipped = (min_x < 0) or (max_x > width) or (min_y < 0) or (max_y > height)
        
    return is_clipped, is_squashed

def get_affine_angle(matrix):
    """
    Calculates the corner angle of an affine transform.
    Input: A 2x3 or 2x2 affine matrix (numpy array or list of lists)
    Output: valid, angle (bool if the transform is close enough to a rectangel)
    """
    # Ensure it's a numpy array
    valid = False
    m = np.array(matrix)
    
    u = m[0:2, 0]  # [a, c]
    v = m[0:2, 1]  # [b, d]
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return False, 180
    cos_theta = dot_product / (norm_u * norm_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    #check if it's square-ish
    if angle_deg - 90 == 0.0:
        valid = False
        print(f'too perfect 0.0')
        return valid, 180
    elif abs(angle_deg - 90) < 3:
        valid = True
        print(f'within 3 degrees :{angle_deg - 90}')
    else:
        valid = False
        print(f'not within 3 degrees :{angle_deg - 90}')
        
    return valid, abs(angle_deg - 90)

def computeAffineTransform(x1, x2):
    n = x1.shape[0]
    
    ## we need at least 3 points for affine transformation so ned to check here
    if n < 3:
        return None
    
    ## convert to homogeneous coordinates
    x1H = np.hstack([x1, np.ones((n, 1))])

    
    ## build matrix M (2n x 6) and vector b (2n x 1)
    M = np.zeros((2*n, 6))
    b = np.zeros((2*n, 1))
    
    for i in range(n):
        ## first row for x-coordinate
        M[2*i, 0:3] = x1H[i]
        ## second row for y-coordinate  
        M[2*i+1, 3:6] = x1H[i]
        
        b[2*i] = x2[i, 0]          # x2_i
        b[2*i+1] = x2[i, 1]        # y2_i
    
    ## solve the linear system: M * params = b
    ## params === [a, b, tx, c, d, ty]^T
    params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    params = params.flatten()
    
    ## build affine matrix
    A = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0,         0,         1        ]
    ])
    
    return A

def paramsToAffine(p):
    """
    Convert 6 parameters to 3x3 affine transformation matrix with this method so i dont have to redo 
    
    p = [p1, p2, p3, p4, p5, p6]
    
    A = | p1+1  p2    p3 |
        | p4    p5+1  p6 |
        | 0     0     1  |

    remember that the +1 is needed because
    """
    A = np.array([
        [p[0] + 1, p[1], p[2]],
        [p[3], p[4] + 1, p[5]],
        [0, 0, 1]
    ])

    return A

## TODO: duplacate this for take or make a scene or take input
def extractSceneTextFromFrames(warped_frames, model_name, showScanBoxes):
    '''
    Extract text from warped slate frames using TrOCR
    Extracts text from the top middle portion of each frame
    Image is divided into 6 parts: horizontally in half, top half split into 3 columns
    where middle column is 50% of the width
    
    Parameters:
    - warped_frames (list): List of warped grayscale frames as numpy arrays
    - model_name (str): TrOCR model to use for text extraction
    - showScanBoxes (bool): If True, display frames with bounding boxes around scan regions
    
    Returns:
    - extracted_texts (list): List of extracted text strings, one per frame
    '''
    
    print(f'Loading TrOCR model: {model_name}...\n');
    
    processor = TrOCRProcessor.from_pretrained(model_name);
    model = VisionEncoderDecoderModel.from_pretrained('agomberto/trocr-large-handwritten-fr')
    tokenizer = AutoTokenizer.from_pretrained('agomberto/trocr-large-handwritten-fr')

    
    extracted_texts = [];
    
    if showScanBoxes:
        num_frames = len(warped_frames);
        cols = min(4, num_frames);
        rows = (num_frames + cols - 1) // cols;
        plt.figure(figsize=(cols * 4, rows * 3));
    
    for i, warped_frame in enumerate(warped_frames):
        ## Get frame dimensions
        h, w = warped_frame.shape;
        
        ## Split horizontally in half
        top_half_height = h // 2;
        
        ## Split top half into 3 columns: left (25%), middle (50%), right (25%)
        left_col_width = int(w * 0.30);
        middle_col_width = int(w * 0.40);
        
        ## Calculate boundaries for top middle region
        # top_row_start = 0;
        ## make top row start at a little lower
        top_row_start = top_half_height//4
        top_row_end = top_half_height;
        middle_col_start = left_col_width;
        middle_col_end = left_col_width + middle_col_width;
        
        if showScanBoxes:
            # Create a copy for visualization with bounding box
            vis_frame = warped_frame.copy();
            
            # Draw rectangle on the frame (need to convert to BGR for color)
            vis_frame_rgb = cv2.cvtColor(vis_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR);
            cv2.rectangle(vis_frame_rgb, 
                         (middle_col_start, top_row_start), 
                         (middle_col_end, top_row_end), 
                         (255, 0, 0), 20);  # red rectangle, thickness 20
            
            plt.subplot(rows, cols, i + 1);
            plt.imshow(vis_frame_rgb);
            plt.title(f'Frame {i+1} - Scan Region');
            plt.axis('off');
        
        ## Crop to top middle region
        cropped_frame = warped_frame[top_row_start:top_row_end, middle_col_start:middle_col_end];
        
        ## Convert numpy array to PIL Image
        ## Ensure the frame is in uint8 format
        frame_uint8 = np.clip(cropped_frame, 0, 255).astype(np.uint8);
        #image = Image.open(cropped_frame).convert("RGB")
        #image = Image.open(cropped_frame).convert("RGBA")
        
        ## Convert grayscale to RGB (TrOCR apparently expects RGB)
        pil_image = Image.fromarray(frame_uint8).convert('RGB');
        #background = Image.new("RGBA", image.size, (255, 255, 255))
        #combined = Image.alpha_composite(background, image).convert("RGB")

        ## Process image for TrOCR
        #pixel_values = processor(pil_image, return_tensors='pt').pixel_values;
        #pixel_values = processor(combined, return_tensors="pt").pixel_values
        
        ## Generate text
        pixel_values = (processor(images=pil_image, return_tensors="pt").pixel_values)
        generated_ids = model.generate(pixel_values)
        extracted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #generated_ids = model.generate(pixel_values)
        #extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #generated_ids = model.generate(pixel_values);
        #extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0];
        
        extracted_texts.append(extracted_text);

        print(f'  Frame {i+1}/{len(warped_frames)}: "{extracted_text}"');
    
    extracted_texts = processExtractedTexts(extracted_texts);
    
    # Show visualization if requested
    if showScanBoxes:
        plt.tight_layout();
        plt.show();
    
    return extracted_texts;

def extractTakeTextFromFrames(warped_frames, model_name, showScanBoxes):
    '''
    Extract text from warped slate frames using TrOCR
    Extracts text from the top right portion of each frame
    Image is divided into 6 parts: horizontally in half, top half split into 3 columns
    where middle column is 50% of the width
    
    Parameters:
    - warped_frames (list): List of warped grayscale frames as numpy arrays
    - model_name (str): TrOCR model to use for text extraction
    - showScanBoxes (bool): If True, display frames with bounding boxes around scan regions
    
    Returns:
    - extracted_texts (list): List of extracted text strings, one per frame
    '''
    
    print(f'Loading TrOCR model: {model_name}...\n');
    
    processor = TrOCRProcessor.from_pretrained(model_name);
    model = VisionEncoderDecoderModel.from_pretrained('agomberto/trocr-large-handwritten-fr')
    tokenizer = AutoTokenizer.from_pretrained('agomberto/trocr-large-handwritten-fr')

    
    extracted_texts = [];
    
    if showScanBoxes:
        num_frames = len(warped_frames);
        cols = min(4, num_frames);
        rows = (num_frames + cols - 1) // cols;
        plt.figure(figsize=(cols * 4, rows * 3));
    
    for i, warped_frame in enumerate(warped_frames):
        ## Get frame dimensions
        h, w = warped_frame.shape;
        
        ## Split horizontally in half
        top_half_height = h // 2;
        
        ## Split top half into 3 columns: left (25%), middle (50%), right (25%)
        left_col_width = int(w * 0.30);
        middle_col_width = int(w * 0.40);
        right_col_width = int(w * 0.30);
        
        ## Calculate boundaries for top middle region
        #top_row_start = 0;
        top_row_start = top_half_height//4
        top_row_end = top_half_height;
        middle_col_start = left_col_width;
        middle_col_end = left_col_width + middle_col_width;
        right_col_start = middle_col_end;
        right_col_end = left_col_width + middle_col_width + right_col_start;
        
        if showScanBoxes:
            # Create a copy for visualization with bounding box
            vis_frame = warped_frame.copy();
            
            # Draw rectangle on the frame (need to convert to BGR for color)
            vis_frame_rgb = cv2.cvtColor(vis_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR);
            cv2.rectangle(vis_frame_rgb, 
                         (right_col_start, top_row_start), 
                         (right_col_end, top_row_end), 
                         (255, 0, 0), 20);  # red rectangle, thickness 3
            
            plt.subplot(rows, cols, i + 1);
            plt.imshow(vis_frame_rgb);
            plt.title(f'Frame {i+1} - Scan Region');
            plt.axis('off');
        
        ## Crop to top middle region
        cropped_frame = warped_frame[top_row_start:top_row_end, right_col_start:right_col_end];
        
        ## Convert numpy array to PIL Image
        ## Ensure the frame is in uint8 format
        frame_uint8 = np.clip(cropped_frame, 0, 255).astype(np.uint8);
        #image = Image.open(cropped_frame).convert("RGB")
        #image = Image.open(cropped_frame).convert("RGBA")
        
        ## Convert grayscale to RGB (TrOCR apparently expects RGB)
        pil_image = Image.fromarray(frame_uint8).convert('RGB');
        #background = Image.new("RGBA", image.size, (255, 255, 255))
        #combined = Image.alpha_composite(background, image).convert("RGB")

        ## Process image for TrOCR
        #pixel_values = processor(pil_image, return_tensors='pt').pixel_values;
        #pixel_values = processor(combined, return_tensors="pt").pixel_values
        
        ## Generate text
        pixel_values = (processor(images=pil_image, return_tensors="pt").pixel_values)
        generated_ids = model.generate(pixel_values)
        extracted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #generated_ids = model.generate(pixel_values)
        #extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #generated_ids = model.generate(pixel_values);
        #extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0];
        
        extracted_texts.append(extracted_text);

        print(f'  Frame {i+1}/{len(warped_frames)}: "{extracted_text}"');
    
    extracted_texts = processExtractedTextsTake(extracted_texts);

    
    # Show visualization if requested
    if showScanBoxes:
        plt.tight_layout();
        plt.show();
    
    return extracted_texts;

def processExtractedTextsTake(extractedTexts):
    '''
    Process extracted texts to clean up and format as needed. For noe it will make all text uppercase and remove spaces and periods.
    
    Parameters:
    - extractedTexts (list): List of raw extracted text strings
    
    Returns:
    - processedTexts (list): List of cleaned/formatted text strings
    '''
    
    processedTexts = [];
    
    for text in extractedTexts:
        text = text.upper().strip();
        text = text.replace(" ", "");
        text = text.replace(".", "");
        text = re.sub(r'\D', '', text)
        #this will remove non numbers

        processedTexts.append(text);
    
    return processedTexts

def processExtractedTexts(extractedTexts):
    '''
    Process extracted texts to clean up and format as needed. For noe it will make all text uppercase and remove spaces and periods.
    
    Parameters:
    - extractedTexts (list): List of raw extracted text strings
    
    Returns:
    - processedTexts (list): List of cleaned/formatted text strings
    '''
    
    processedTexts = [];
    
    for text in extractedTexts:
        text = text.upper().strip();
        text = text.replace(" ", "");
        text = text.replace("'", "");
        text = text.replace(".", "");

        processedTexts.append(text);
    
    return processedTexts

'''clean the text'''
def cleanScene(s, debug = False):
    match = re.search(r"\d+[A-Z]", s) #\d+[A-Z] #\d+[A-Za-z]
    #catches error when theres no capital Letter w/ num
    print(f'match : {match}')
    if not match:
        if debug:
            print("scene is not parsable returning none")
        return None
    return match[0]

def findScene(results):
    # print(f'x[0] for x in results, {(x[0] for x in results)}')
    db = (x for x in results)
    # print(db)
    #cleanSold = [cleanScene(x[0]) for x in results]
    #cleanS = cleanScene(results)
    pattern = r"^\d+[A-Z]$"
    #finds most common one (use Count dict) FUTURE: maybe combine w/ findTake
    # print(f'results {results}')
    sceneCountDict = Counter(results)
    print(sceneCountDict)
    modeScene = sceneCountDict.most_common(1)
    if modeScene[0][1] == 1:
        print('mode is 1 data point not confident')
        return 'None'
    #print(cleanS)
    print(modeScene)
    sceneRes = modeScene[0][0]
    
    if bool(re.match(pattern, sceneRes)):
        return(sceneRes)
    else:
        print('scene result is not in corect format')
        return 'None'

    #return(sceneRes)

'''calculate the predicted value of the take based on the mode of what was read'''
def findTake(results):
    #finds most common one
    takeCountDict = Counter(results)
    modeTake = takeCountDict.most_common(1)
    #TODO: edge case when mode is a tie and or low mode
    #print(cleanT)
    #print(modeTake)
    takeRes = modeTake[0][0]
    if takeRes.isdigit():
        return(takeRes)
    else:
        return 'None'

def renameVid(folder_path, filename, scene, take, debug=False):
    
    # 1. Construct the full path to the EXISTING file
    # We combine the folder path with the filename to find it
    old_full_path = os.path.join(folder_path, filename)
    
    # 2. Construct the full path for the NEW name
    # We create the new name string (scene.take.filename)
    # And join it with the SAME folder path so it stays there
    new_name = f"{scene}.{take}.{filename}"
    new_full_path = os.path.join(folder_path, new_name)
    
    if debug:
        print(f"Renaming inside {folder_path}:")
        print(f"From: {filename}")
        print(f"To:   {new_name}")

    # 3. Perform the rename

    os.rename(old_full_path, new_full_path)

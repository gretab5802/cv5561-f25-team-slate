'''
Docstring for helperfunctions

helper functions included in this file:

- extractVidFrames
- extract_frames_at_interval

'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

def getTemplateImage():
    template = Image.open('template.jpeg');
    template = np.array(template.convert('L'));
    return template;

def getVideoFile(filepath):
    video_file = filepath;
    return video_file;

def extractFrames(video_filepath, frame_interval):
    '''
    Extract frames from a video at specified intervals.
    
    Parameters:
    - video_filepath (str): Path to the video file
    - frame_interval (int): Extract every Nth frame (e.g., 6 means extract every 6th frame)
    
    Returns:
    - extracted_frames (list): List of extracted frames as numpy arrays
    '''
    
    extracted_frames = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_filepath)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filepath}\n")
        return extracted_frames
    
    frame_count = 0
    
    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If frame reading was not successful, break
        if not ret:
            break
        
        # Extract frame if it's at the specified interval
        if frame_count % frame_interval == 0:
            extracted_frames.append(frame)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    return extracted_frames

def find_match(img1, img2):
    x1, x2 = None, None
    dis_thr = 0.7
    
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
    neighbors_back = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    distances_back, indices_back = neighbors_back.kneighbors(des2)

    good_matches = []

    for i in range(len(distances)):
        dis_closest = distances[i][0]
        dis_second = distances[i][1]
        
        if dis_second > 0 and (dis_closest / dis_second) < dis_thr:
            j = indices[i][0]  ## best match in img2

            ## apply Lowe's ratio test for backward match as well
            dis_back_closest = distances_back[j][0]
            dis_back_second = distances_back[j][1]
            
            ## cross-check: verify that img2[j]'s best match is img1[i]
            ## AND that backward match also passes ratio test
            if (indices_back[j][0] == i and 
                dis_back_second > 0 and 
                (dis_back_closest / dis_back_second) < dis_thr):
                good_matches.append((i, j))
    
    x1 = np.zeros((len(good_matches), 2))
    x2 = np.zeros((len(good_matches), 2))

    for idx, (i, j) in enumerate(good_matches):
        x1[idx] = kp1[i]
        x2[idx] = kp2[j]

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = None

    n = x1.shape[0]
    
    ## need at least 3 points for affine transform
    if n < 3:
        return np.eye(3)
    
    best_inliers = []
    best_A = None
    
    ## RANSAC iterations
    for iteration in range(ransac_iter):
        ## randomly sample 3 points
        ## np.random.choice returns indices!!!
        indices = np.random.choice(n, size=3, replace=False)
        sample_x1 = x1[indices]
        sample_x2 = x2[indices]
        
        ## compute affine transform from these 3 points
        A = compute_affine_transform(sample_x1, sample_x2)
        
        if A is None:
            continue
        
        ## transform all x1 points using this affine matrix
        ## convert x1 to homogeneous coordinates (n x 3)
        x1_h = np.hstack([x1, np.ones((n, 1))])
        
        ## transform: x2_pred = A * x1
        ## x2_pred will be (n x 3), we take first 2 columns
        x2_pred = (A @ x1_h.T).T
        x2_pred = x2_pred[:, :2]  ## remove homogeneous coordinate
        
        ## compute errors (Euclidean distance)
        ## errors[i] = distance between predicted and actual point
        errors = np.sqrt(np.sum((x2_pred - x2)**2, axis=1))
        
        ## find inliers (points with error < threshold)
        inliers = np.where(errors < ransac_thr)[0]
        
        ## keep track of best model (one with most inliers)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_A = A
    
    ## refine the affine transform using ALL inliers
    ## this gives better accuracy than using just 3 random points
    if len(best_inliers) >= 3:
        best_A = compute_affine_transform(x1[best_inliers], x2[best_inliers])
    
    ## ff RANSAC failed to find any good model, just return identity - reminder to me that this means error and need to check somethign 
    if best_A is None:
        best_A = np.eye(3)

    return best_A

def warp_image(img, A, output_size):
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

    ## print("output size")
    ## print(np.meshgrid(np.arange(w_out), np.arange(h_out))[0].shape)

    ## print("img_warped size")
    ## print(img_warped.shape)

    return img_warped

def align_image(template, target, A):
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
        A_current = params_to_affine(p)
        
        img_warped = warp_image(target, A_current, template.shape)
        
        I_err = img_warped - template
        
        ## error
        error_norm = np.sqrt(np.mean(I_err ** 2)) / 255.0  ## normalize by 255
        errors.append(error_norm)
        
        I_err_flat = I_err.flatten()
        F = steepest_descent.T @ I_err_flat
        
        delta_p = np.linalg.solve(H, F)  ## More stable than inv(H) @ F
        
        A_delta = params_to_affine(delta_p)
        
        ## invert A_delta
        A_delta_inv = np.linalg.inv(A_delta)
        
        A_new = A_current @ A_delta_inv
        
        ## extract parameters from new affine matrix
        p = np.array([A_new[0, 0] - 1, A_new[0, 1], A_new[0, 2],
                      A_new[1, 0], A_new[1, 1] - 1, A_new[1, 2]])
        
        # check with what we set epsilon as === 0.0001
        if np.linalg.norm(delta_p) < epsilon:
            break
    
    A_refined = params_to_affine(p)
    errors = np.array(errors)
    
    return A_refined, errors

def track_multi_frames(template, img_list, breakProcessingEarly):
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
    ransac_thr = 3.0
    ransac_iter = 1000
    
    A_list = []
    errors_list = []
    
    original_template = template.copy()
    
    ## current template (will be updated after each frame)
    current_template = template.copy()
    
    ## initial Affine matrix
    A_prev = None
    
    for i, target_img in enumerate(img_list):

        if breakProcessingEarly and i >= 10:
            print("Breaking at frame 10 for testing purposes.")
            break

        print(f"Processing frame {i+1}/{len(img_list)}...")
        
        if i == 0:
            ## FIRST FRAME: Use feature matching + RANSAC
            print("  Using feature matching for initialization...")
            
            x1, x2 = find_match(current_template, target_img)
            
            A_init = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
            
        else:
            ## after initial frame, use previous A as init
            A_init = A_prev.copy()
        
        ## refine with ICIA
        A_refined, errors = align_image(current_template, target_img, A_init)
        
        A_list.append(A_refined)
        errors_list.append(errors)
        
        ## update the template for next frame
        if i < len(img_list) - 1:  ## Don't need to update after last frame
            current_template = warp_image(target_img, A_refined, current_template.shape)

            ## formattinsg
            # current_template = np.clip(current_template, 0, 255).astype(np.uint8)
        
        ## save current affine for next frame's initialization
        A_prev = A_refined.copy()

    return A_list, errors_list

def compute_affine_transform(x1, x2):
    n = x1.shape[0]
    
    ## we need at least 3 points for affine transformation so ned to check here
    if n < 3:
        return None
    
    ## convert to homogeneous coordinates
    x1_h = np.hstack([x1, np.ones((n, 1))])

    
    ## build matrix M (2n x 6) and vector b (2n x 1)
    M = np.zeros((2*n, 6))
    b = np.zeros((2*n, 1))
    
    for i in range(n):
        ## first row for x-coordinate
        M[2*i, 0:3] = x1_h[i]
        ## second row for y-coordinate  
        M[2*i+1, 3:6] = x1_h[i]
        
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

def params_to_affine(p):
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

def extract_text_from_frames(warped_frames, model_name):
    '''
    Extract text from warped slate frames using TrOCR
    
    Parameters:
    - warped_frames (list): List of warped grayscale frames as numpy arrays
    - model_name (str): TrOCR model to use for text extraction
    
    Returns:
    - extracted_texts (list): List of extracted text strings, one per frame
    '''
    
    print(f'Loading TrOCR model: {model_name}...\n');
    
    processor = TrOCRProcessor.from_pretrained(model_name);
    model = VisionEncoderDecoderModel.from_pretrained(model_name);
    
    extracted_texts = [];
    
    print('Extracting text from warped frames...\n');
    
    for i, warped_frame in enumerate(warped_frames):
        # Convert numpy array to PIL Image
        # Ensure the frame is in uint8 format
        frame_uint8 = np.clip(warped_frame, 0, 255).astype(np.uint8);
        
        # Convert grayscale to RGB (TrOCR apparently expects RGB)
        pil_image = Image.fromarray(frame_uint8).convert('RGB');
        
        # Process image for TrOCR
        pixel_values = processor(pil_image, return_tensors='pt').pixel_values;
        
        # Generate text
        generated_ids = model.generate(pixel_values);
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0];
        
        extracted_texts.append(extracted_text);
        print(f'  Frame {i+1}/{len(warped_frames)}: "{extracted_text}"');
    
    return extracted_texts;
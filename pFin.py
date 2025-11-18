from PIL import Image
import cv2 as cv
import numpy as np
from cv2 import resize
import matplotlib.pyplot as plt
import sklearn

from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

import cv2

# Get frames of the video(s)
def extractVidFrames(filename, start_frame=0, end_frame=-1, skip=10, debug = False):
    results = None
    cap = cv2.VideoCapture(filename)
    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"fps: {fps}!")
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
        ret, frame = cap.read()

        # If the frame is empty, break out of the loop
        if not ret:
            break
        cap.release()
        print("cap release")
        return results
    


# for each frame run sift find slate (check if thats a reasonable slate representation)
# find more slates from neighboring frames...?

# find text in the slate

# find the label



# def extractVidFrames(filename, start_frame=0, end_frame=-1, skip=10, debug = False):
#     cap = cv2.VideoCapture(filename)

#     # Get the frame rate
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     #print(f"fps: {fps}!")
#     # Calculate the interval between frames
#     #interval = 1 / fps

#     # Create a counter to keep track of the frame number
#     frame_count = 0
#     #keeps track of most recent frame with slate identified
#     captFrame = 0
#     #print(cap.isOpened())
#     cap.isOpened() == True
#     results = []
#     # Loop through the frames
#     while cap.isOpened():

#         # Capture the next frame
#         ret, frame = cap.read()

#         # If the frame is empty, break out of the loop
#         if not ret:
#             break

#         # Increment the frame counter
#         frame_count += 1

#         # Save the frame as an image
#         if frame_count >= start_frame and frame_count%skip == 0 and ((frame_count <= end_frame) or (end_frame == -1)):
#             if debug:
#                 print(f"frame num: {frame_count}!")
#                 #print(f"last captured frame num: {captFrame}!")
#                 cv2.imwrite('frame_{}.jpg'.format(frame_count), frame)
#                 frameFile = ('frame_{}.jpg'.format(frame_count))
#             esi = extractSlateImg(frameFile)#matrix representing the image itself an ndarray
#             #print(esi)
#             #print(type(esi))
#             if isinstance(esi, np.ndarray):
#                 st = extractSceneAndTake(esi, debug)
#                 captFrame = frame_count
#                 if debug:
#                     cv2.imwrite('slate_{}.jpg'.format(frame_count), esi)
#                     print(f"Found scene/take: {st}")
#                 results.append(st)
#             elif len(results) > 6:
#                 if debug:
#                     print("found at least 6")
#                     print(f"results: {results}")
#                 cap.release()
#                 return results
#             #print(f"frame_count - captFrame = {frame_count - captFrame}")
#             #print(f"length of results: {len(results)}")
#             if (len(results) >= 1) and ((frame_count - captFrame) > 200):
#                 if debug:
#                     print("found at least 1, but now the slate is gone")
#                     print(f"results: {results}")
#                 cap.release()
#                 print("cap release")
#                 return results
#             if (len(results) == 0) and ((frame_count - captFrame) > 500):
#                 if debug:
#                     print("found none, can't find slate")
#                     print(f"results: {results}")
#                 cap.release()
#                 print("cap release")
#                 return results

#             #elif esi == 0:

#     #make it stop watching at a threshold
#     # Release the video capture object
#     if debug:
#         print("went through entire video")
#         print(f"results: {results}")
#     cap.release()
#     return results


def find_match(img1, img2):
    x1, x2 = None, None
    dis_thr = .7

    sift = SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None) #points are similar if the descriptprs are similar
    kp2, des2 = sift.detectAndCompute(img2,None)
    kpAH = KeyPoint_convert(kp1)
    kpBH = KeyPoint_convert(kp2)
    print(f'key points are: {kpAH}, \n {kpBH}')
    # compare each keypoint forward and backward by descriptior to find the most similar
    # this just finds neighbors in a set we need it across the sets

    nbrs2 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)

    # Use the kneighbors method to find the nearest neighbor
    # It returns the distances and the indices of the neighbors
    distancesA, indicesAtoB = nbrs2.kneighbors(des1) # indiciesAtoB means the index indiciesAtoB is an image A value and the value at that index is the image B value.
    ratioA = distancesA[:,0] / distancesA[:,1]

    nbrs1 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    distancesB, indicesBtoA = nbrs1.kneighbors(des2)
    ratioB = distancesB[:,0] / distancesB[:,1]
    print(f'indices: {indicesAtoB}, {indicesBtoA}')

    kp2HKeep = []
    kp1HKeep = []
    # Get the nearest neighbor from the target dataset
    for i in range(len(distancesA)):
        if ratioA[i] < dis_thr:
            nearest_neighbor_indexB = indicesAtoB[i][0]
            nearest_neighbor_pointB = kpBH[nearest_neighbor_indexB]
            nearest_neighbor_pointA = kpAH[i]
            if indicesBtoA[nearest_neighbor_indexB][0] == i:
                if ratioB[nearest_neighbor_indexB] < dis_thr:
                    kp2HKeep.append(nearest_neighbor_pointB)
                    kp1HKeep.append(nearest_neighbor_pointA)
                
    # nearest neighbor
    x1 = np.array(kp1HKeep)
    x2 = np.array(kp2HKeep)
    #   x1      x2
    # [x,y] - [x,y]
    # . . . . 
    # each row is a matching pair

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = 0

    # To do
    Esums = []
    modelIndicies = []
    #affine transform
    for i in range(ransac_iter):
        fail = 0
        indicies = []
        indicies.append(np.random.randint(0, len(x1), size=None))
        while fail == 0:
            new = np.random.randint(0, len(x1), size=None)
            if new != indicies[0]:
                indicies.append(new)
                fail = 1

        while fail == 1:
            new2 = np.random.randint(0, len(x1), size=None)
            if new2 != indicies[0] and new2 != indicies[1]:
                indicies.append(new2)
                fail = 2

        orig = np.array([x1[indicies[0]],x1[indicies[1]],x1[indicies[2]]])


        if np.linalg.cond(np.hstack((orig, np.ones((3,1))))) > 100000:
            continue

        dest = np.array([x2[indicies[0]],x2[indicies[1]],x2[indicies[2]]]) #there seem to be some duplicates at diff indicies in x2
        #get s points
        oriMat = np.zeros((6, 6))
        for j in range(3):
            oriMat[j, 0:3] = [orig[j, 0], orig[j, 1], 1]
            oriMat[j+3, 3:6] = [orig[j, 0], orig[j, 1], 1]

        desMat = np.zeros((6, 1))
        for j in range(3):
            desMat[j, 0] = dest[j, 0]
            desMat[j+3, 0] = dest[j, 1]

        params = np.linalg.solve(oriMat, desMat)
        
        Atemp = np.array([[params[0, 0], params[1, 0], params[2, 0]],
                          [params[3, 0], params[4, 0], params[5, 0]],
                          [0.0, 0.0, 1.0]])

        # count inliers, record
        x1s = np.concatenate((x1, np.ones((len(x1), 1))), axis = 1)
        x2s = np.concatenate((x2, np.ones((len(x2), 1))), axis = 1)

        x1Trans = Atemp @ x1s.T
        errors = x1Trans - x2s.T
        normErr = np.sqrt(np.sum(errors[:2]**2, axis = 0))
        outliers = [1 for item in normErr if abs(item) > ransac_thr]
        errorCount = sum(outliers)
        
        Esums.append(errorCount)
        if i == 0:
            A = Atemp
        elif Esums[len(Esums)-1] == min(Esums):
            A = Atemp

    # do it ransac_iter times over
    # find and report A from the best model
    return A

def warp_image(img, A, output_size):
    img_warped = None
    width = img.shape[0]
    height = img.shape[1]
    points = (np.arange(height), np.arange(width))
    allInvC = []
    #loop throught x and y of output size
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            #find inverse cord
            invCord = A @ np.array([j, i, 1])
            allInvC.append(invCord[0:2])
    flat = interpolate.interpn(points = points, values = img.T, xi = allInvC, method= 'linear', fill_value=0, bounds_error=False)

    img_warped = flat.reshape(*output_size)
    #points is a grid of origonal image size
    #values is orig image
    #xi is the backwards mapping
    return img_warped



def getParamAffineTrans(p):
    return np.array([[p[0]+1, p[1], p[2]],
                    [p[3], p[4]+1, p[5]],
                    [0.0, 0.0, 1.0]])

def filter_image(image, filter):
    x, y = image.shape
    image_filtered = np.zeros(image.shape)
    image_padded = np.pad(image, 1)
    for i in range(1, x+1):
        for j in range(1, y+1):
            image_filtered[i-1, j-1] = sum(sum((image_padded[i-1:i+2, j-1:j+2]) * (filter)))
    return image_filtered

def get_differential_filter():
    filter_x, filter_y = None, None
    #start
    #sobel filter
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return -filter_x, -filter_y

def align_image(template, target, A):
    A_refined = None
    errors = []
    template = template/255
    target = target/255
    e = .028
    print('aligning image')
    #Compute the gradient of template image, ∇Itpl
    x_filter, y_filter = get_differential_filter()
    image_dx = filter_image(template, x_filter)
    image_dy = filter_image(template, y_filter)
    #Iy, Ix = np.gradient(template)
    H, W = template.shape
    y_coords, x_coords = np.indices((H, W))
    #Compute the Jacobian ∂W/∂p at (x; 0).
    #[[u, v, 1, 0, 0, 0],
    #[0, 0, 0, u, v, 1]]
    steDecnt = np.zeros((H, W, 6))
    # [Ix*x, Ix*y, Ix, Iy*x, Iy*y, Iy]
    steDecnt[:, :, 0] = image_dx * x_coords
    steDecnt[:, :, 1] = image_dx * y_coords
    steDecnt[:, :, 2] = image_dx
    steDecnt[:, :, 3] = image_dy * x_coords
    steDecnt[:, :, 4] = image_dy * y_coords
    steDecnt[:, :, 5] = image_dy
    #Compute the steepest descent images ∇Itpl ∂W/∂p
    #ompute the 6 × 6 Hessian H= (see notes)
    steDecnt_T = steDecnt.reshape(steDecnt.shape[0], steDecnt.shape[1], 6, 1)
    steDecntW1 = np.transpose(steDecnt_T, (0, 1, 3, 2))
    Hess = np.sum(steDecnt_T@steDecntW1, axis=(0,1))
    loopNum = 0
    pMult = 1
    #while True do
    A_refined = A
    err = 0
    print('enter while loop')
    while True:
        warped = warp_image(target, A_refined, [H, W])
        Ierr = warped - template
        Ierr_expanded = Ierr[:, :, np.newaxis] 
        errors.append(np.linalg.norm(Ierr, ord=2))
        F_terms = steDecnt * Ierr_expanded
        F = np.sum(F_terms, axis=(0, 1))
        #Warp the target to the template domain Itgt(W (x; p)).
        #   Compute the error image Ierr = Itgt(W (x; p)) − Itpl.
        #   Compute F = (see notes)
        #multiply p by diff num to control step size 
        deltaP = np.linalg.solve(Hess, F)

        if loopNum < 20:
            pMult = 200
        elif loopNum < 50:
            pMult = 100
        elif loopNum < 100:
            pMult = 50
        elif loopNum < 150:
            pMult = 10
        elif loopNum < 200:
            pMult = 2
        elif loopNum < 300:
            pMult = 1
        elif loopNum < 400:
            pMult = .5
        elif loopNum < 500:
            pMult = .2
        else:
            pMult = .05

        p = getParamAffineTrans(deltaP*pMult)
        print(f'p {p}')
        invP = np.linalg.inv(p)
        A_refined = A_refined @ (invP)

        # based on itteration
        print(f'loop {loopNum}')
        loopNum +=1
        print(f'thresh {np.linalg.norm(deltaP, ord=2)}')
        if np.linalg.norm(deltaP, ord=2) < e:
            # use getParamAffineTrans(p) to update W (p)
            break
    return A_refined, errors

def track_multi_frames(template, img_list):
    A_list = []
    errors_list = []
    ransac_thr = 45 
    ransac_iter = 1000
    for i in range(len(img_list)):
        x1, x2 = find_match(template, img_list[i])
        A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
        A_refined, errors = align_image(template, img_list[i], A)
        A_list.append(A_refined)
        errors_list.append(errors)
    return A_list, errors_list

# ----- Visualization Functions -----
def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    import cv2
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list, errors_list=None):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

    if errors_list is not None:
        for i, errors in enumerate(errors_list):
            plt.plot(errors * 255)
            plt.title(f'Frame {i}')
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.show()
# ----- Visualization Functions -----

if __name__=='__main__':

    template = Image.open('templatefull.jpeg')
    template = np.array(template.convert('L'))
    
    target_list = []
    #target = Image.open(f'target{i}.jpg')
    #target = np.array(target.convert('L'))
    #target_list.append(target)
    for i in range(4):
        target = Image.open(f'target{i+1}.jpeg')
        target = np.array(target.convert('L'))
        target_list.append(target)
    
    x1, x2 = find_match(template, target_list[0])
    print(f'x1, x2: {x1} .\n.\n. {x2}')
    visualize_find_match(template, target_list[0], x1, x2)

    # To do
    ransac_thr = 45
    ransac_iter = 1000
    # ----------
    
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list, errors_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list, errors_list)
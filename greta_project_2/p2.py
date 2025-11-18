from PIL import Image
import numpy as np
from cv2 import resize
import matplotlib.pyplot as plt

from cv2 import SIFT_create, KeyPoint_convert, filter2D
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def find_match(img1, img2):
    x1, x2 = None, None
    dis_thr = 0.7
    
    # To do

    # SIFT feature extraction on both images, keypoints and descriptors for img1 and 2
    sift = SIFT_create()
    kps1, desc1 = sift.detectAndCompute(img1, None)
    kps2, desc2 = sift.detectAndCompute(img2, None)

    # if no features found
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        x1 = np.zeros((0, 2), dtype=np.float32)
        x2 = np.zeros((0, 2), dtype=np.float32)
        return x1, x2

    # Nearest-neighbor search (2 nearest neighbors)
    nn_12 = NearestNeighbors(n_neighbors=2).fit(desc2)
    dists_12, idxs_12 = nn_12.kneighbors(desc1)

    # cross-check consistency
    nn_21 = NearestNeighbors(n_neighbors=2).fit(desc1)
    dists_21, idxs_21 = nn_21.kneighbors(desc2)

    # Mutual (cross-checked) ratio-test matches
    matches = []
    for i in range(len(desc1)):
        if dists_12[i, 0] < dis_thr * dists_12[i, 1]:
            j = idxs_12[i, 0]  # best match in img2 for descriptor i in img1
            # Cross-check if img2's best for j return to i
            if idxs_21[j, 0] == i and dists_21[j, 0] < dis_thr * dists_21[j, 1]:
                matches.append((i, j))

    # Convert matched keypoints
    all_kp1 = KeyPoint_convert(kps1)
    all_kp2 = KeyPoint_convert(kps2)

    x1 = all_kp1[[i for i, _ in matches]].astype(np.float32)
    x2 = all_kp2[[j for _, j in matches]].astype(np.float32)

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    A = None

    # To do

    # Require <= 3 point pairs
    if x1 is None or x2 is None or x1.shape[0] < 3 or x2.shape[0] < 3:
        # Fall back to identity if not enough
        A = np.eye(3, dtype=np.float32)
        return A

    num_pts = x1.shape[0]
    best_inlier_count = 0
    A = np.eye(3, dtype=np.float32)

    # Homogeneous coordinates
    x1_h = np.hstack([x1, np.ones((num_pts, 1), dtype=np.float32)])
    x2_h = np.hstack([x2, np.ones((num_pts, 1), dtype=np.float32)])

    # RANSAC: randomly sample 3 correspondences, score, keep best
    for _ in range(ransac_iter):
        sample_idx = np.random.choice(num_pts, 3, replace=False)
        A_candidate = compute_affine(x1[sample_idx], x2[sample_idx])  # LSQ fit on sample

        # Project all x1 points with candidate model
        proj = (A_candidate @ x1_h.T).T
        proj /= proj[:, 2:3]  # normalize homogeneous coordinate

        # pixel errors in the image plane
        errs = np.linalg.norm(x2_h[:, :2] - proj[:, :2], axis=1)
        inliers = errs < ransac_thr
        inlier_count = int(np.sum(inliers))

        # keep best
        if inlier_count > best_inlier_count and inlier_count >= 3:
            best_inlier_count = inlier_count
            A = compute_affine(x1[inliers], x2[inliers])

    return A


def compute_affine(x1, x2):
    n = x1.shape[0]
    A = np.zeros((2 * n, 6))
    b = np.zeros((2 * n, 1))
    A[0::2, 0:2] = x1
    A[1::2, 2:4] = x1
    A[0::2, 4] = 1
    A[1::2, 5] = 1
    b[0::2, 0] = x2[:, 0]
    b[1::2, 0] = x2[:, 1]
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.array([[p[0, 0], p[1, 0], p[4, 0]],
                  [p[2, 0], p[3, 0], p[5, 0]],
                  [0, 0, 1]])
    return M


def warp_image(img, A, output_size):
    img_warped = None

    # To do

    out_h, out_w = output_size
    src_h, src_w = img.shape

    yy, xx = np.meshgrid(np.arange(out_h, dtype=np.float32),
                         np.arange(out_w, dtype=np.float32),
                         indexing='ij')

    # Convert to homogeneous coordinates
    out_homo = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=np.float32)], axis=1)

    # map output pixels into source image coordinates
    # A is affine
    mapped = (A @ out_homo.T).T
    # extract mapped continuous positions in source image
    mapped_x = mapped[:, 0]
    mapped_y = mapped[:, 1]

    src_y_grid = np.arange(src_h, dtype=np.float32)
    src_x_grid = np.arange(src_w, dtype=np.float32)

    sample_points = np.stack([mapped_y, mapped_x], axis=-1)

    # interpolation padding 0s
    img_warped = interpolate.interpn(
        (src_y_grid, src_x_grid),  # axes order (y, x)
        img.astype(np.float32),
        sample_points,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Reshape back to (H, W)
    # had clip to valid byte range
    img_warped = img_warped.reshape(out_h, out_w)
    img_warped = np.clip(img_warped, 0, 255)
    return img_warped


def align_image(template, target, A):
    A_refined = None
    errors = None
    # To do

    # Initialize p = p0 from input A
    A_current = A.copy().astype(np.float32)
    errors = []
    epsilon = 1e-4

    tpl = template.astype(np.float32)
    tgt = target.astype(np.float32)
    if tpl.ndim == 3:
        tpl = tpl.mean(axis=2)
    if tgt.ndim == 3:
        tgt = tgt.mean(axis=2)
    tpl *= (1.0 / 255.0)
    tgt *= (1.0 / 255.0)

    # Compute the gradient of template image, ∇Itp
    dTy, dTx = np.gradient(tpl)  # Iy, Ix

    # Compute the Jacobian ∂W∂p at (x; 0)
    h, w = tpl.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))

    # Compute the steepest descent images ∇Itpl ∂W 
    SD = np.zeros((u.size, 6), dtype=np.float32)
    flat_u, flat_v = u.ravel(), v.ravel()
    flat_Ix, flat_Iy = dTx.ravel(), dTy.ravel()

    # Compute the 6 ×6 Hessian H = ∑x[∇Itpl ∂W∂p]T[∇Itpl ∂W∂p]
    SD[:, 0] = flat_Ix * flat_u  # Ix * u
    SD[:, 1] = flat_Ix * flat_v  # Ix * v
    SD[:, 2] = flat_Ix           # Ix * 1
    SD[:, 3] = flat_Iy * flat_u  # Iy * u
    SD[:, 4] = flat_Iy * flat_v  # Iy * v
    SD[:, 5] = flat_Iy           # Iy * 1

    H = SD.T @ SD               # Hessian H
    H_inv = np.linalg.pinv(H)

    while True:
        # Warp the target to the template domain Itgt(W(x; p))
        warped = warp_image(tgt, A_current, tpl.shape)

        # Compute F = ∑x[∇Itpl ∂W∂p]TIerr
        F = tpl.ravel() - warped.ravel()

        # Compute the error image Ierr = Itgt(W(x; p)) −Itpl
        mse = float(np.mean(F ** 2))
        errors.append(mse)

        # Compute ∆p = H−1F
        delta_p = H_inv @ (SD.T @ F)
        delta_p_norm = float(np.linalg.norm(delta_p))

        # Update W(x; p) ←W(x; p) ◦W−1(x; ∆p) = W(W−1(x; ∆p); p)
        delta_A = np.array([[1.0 + delta_p[0], delta_p[1],        delta_p[2]],
                            [delta_p[3],       1.0 + delta_p[4],  delta_p[5]],
                            [0.0,              0.0,               1.0      ]],
                           dtype=np.float32)

        # A_new = A_old @ ΔA
        A_current = A_current @ delta_A

        # if ∥∆p∥< ε then break
        if delta_p_norm < epsilon:
            break

    # Return A_refined made of p
    A_refined = A_current
    errors = np.asarray(errors, dtype=np.float32)

    return A_refined, errors


def track_multi_frames(template, img_list):
    A_list = None
    errors_list = None

    # To do

    A_list = []
    errors_list = []

    # match keypoints between the template and first frame
    x1, x2 = find_match(template, img_list[0])

    # if not enough matches start with identity transform
    if x1 is None or x2 is None or x1.shape[0] < 3:
        A_init = np.eye(3, dtype=np.float32)
    else:
        # estimate with RANSAC
        ransac_thr = 3.0
        ransac_iter = 500
        A_init = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    A_refined, errs = align_image(template, img_list[0], A_init)
    A_list.append(A_refined)
    errors_list.append(errs)

    # update template
    rolling_template = warp_image(img_list[0], A_refined, template.shape)
    prev_affine = A_refined.copy()

    # align remaining frames sequentially
    for i in range(1, len(img_list)):
        # refine alignment using previous frame as reference
        A_refined, errs = align_image(rolling_template, img_list[i], prev_affine)

        A_list.append(A_refined)
        errors_list.append(errs)

        # update rolling template for next iteration
        rolling_template = warp_image(img_list[i], A_refined, rolling_template.shape)
        prev_affine = A_refined.copy()

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

    template = Image.open('template.jpg')
    template = np.array(template.convert('L'))
    
    target_list = []
    for i in range(4):
        target = Image.open(f'target{i+1}.jpg')
        target = np.array(target.convert('L'))
        target_list.append(target)
    
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    # To do
    ransac_thr = None
    ransac_iter = None
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

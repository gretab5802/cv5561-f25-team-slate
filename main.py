import cv2
import os
import pupil_apriltags
import PIL
import transformers
import sys
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pFin import *


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

print("hello world")

#st = extractSceneAndTake(extractSlateImg('SlateTest1.jpg'))
#print(st)



#TODO: make this iterate through a file
#for filename in sys.argv[1:]:
#    proccessAndRenameVid(filename, True)

template = Image.open('templatefull.jpeg')
template = np.array(template.convert('L'))
#template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

for filename in sys.argv[1:]:
    master(filename, template)


# target_list = []
# for filename in sys.argv[1:]:
#     file = filename
#     #proccessAndRenameVid(filename, True)
# #target = Image.open(f'target{i}.jpg')
# #target = np.array(target.convert('L'))
# #target_list.append(target)

# #do get frames thing here
# extractVidFrames(file)
# for i in range(4):
#     target = Image.open(f'target{i+1}.jpeg')
#     target = np.array(target.convert('L'))
#     target_list.append(target)

# x1, x2 = find_match(template, target_list[0])
# print(f'x1, x2: {x1} .\n.\n. {x2}')
# visualize_find_match(template, target_list[0], x1, x2)

# # To do
# ransac_thr = 45
# ransac_iter = 1000
# # ----------

# A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
# visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr)

# img_warped = warp_image(target_list[0], A, template.shape)
# plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
# plt.axis('off')
# plt.show()

# A_refined, errors = align_image(template, target_list[1], A)
# visualize_align_image(template, target_list[1], A, A_refined, errors)

# A_list, errors_list = track_multi_frames(template, target_list)
# visualize_track_multi_frames(template, target_list, A_list, errors_list)


########################################################################################################################################################
# Note: please install requirements.txt file before testing of the project so that all the libraries used in the project can be installed successfully.
########################################################################################################################################################

# Importing important libraries for the project
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from eval_segm import pixel_accuracy, calculate_iou, calculate_mae

# Load the video
# change this path for testing purpose
# video 1) squat_1667.mp4
# video 2) squat_1668.mp4
# video 3) squat_1677.mp4
video_path = 'squat_1677.mp4'

cap = cv2.VideoCapture(video_path)
output_file_name = video_path.split(".")[0]

if not os.path.isdir('Output'):
    os.mkdir('Output')

# Reading the first frame as image1 and defining image2 as None
_, image1 = cap.read()
image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2 = None

# Obtaining frame height and width
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# dictionary for a perfect frame for each video
perfect_frames = {"squat_1667":41, "squat_1668":45, "squat_1677":40}

# Defining similarity score and frame number
similarity_score = 1
frame_number = None
i = 1

perfect_frame = None
result_frame = None
perfect_frame_number = None


# Loop to find the frame with the highest structural similarity to the first frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if output_file_name in perfect_frames.keys():
        if i == perfect_frames[output_file_name]:
            perfect_frame = frame
            perfect_frame_number = i

    # Calculate Structural Similarity Index (SSI)
    ssi = ssim(image1, frame_gray)

    # Update similarity score and frame number if a better match is found
    if ssi < similarity_score:
        similarity_score = ssi
        result_frame = frame
        image2 = frame_gray
        cv2.imwrite("Output/"+output_file_name+".jpg", frame)
        frame_number = i
    i += 1

# image selection
cv2.imshow("Image selection", np.hstack((perfect_frame, result_frame)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# printing Mean Absolute Error (MAE) for image selection
MAE = calculate_mae(i, perfect_frame_number, frame_number)
print("MAE for the frames:", MAE)

# Reinitialize the video capture object
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
i = 1
highest_contour_area = 0

if output_file_name == "squat_1677":
    threshold = 4
else:
    threshold = 5

# Loop for contour analysis
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze contours to find the one with the highest area
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 10000:
            if cv2.contourArea(contour) > highest_contour_area:
                highest_contour_area = cv2.contourArea(contour)
                cnt = contour

    # Break the loop if the frame number matches the one with the best match
    if i == frame_number:
        img = frame1
        break
    i += 1

    # Read the next frame
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret:
        break

# Create a mask based on the best contour
mask = np.zeros((frame_height, frame_width), dtype="uint8")
cnt = sorted(contours, key=cv2.contourArea)[-1]
maskedRed = cv2.drawContours(mask, [cnt], -1, (0, 0, 255), -1)
maskedFinal = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

# Apply the mask to the original frame
finalImage = cv2.bitwise_and(img, img, mask=maskedFinal)

# alpha = np.sum(finalImage, axis=-1) > 0

# alpha = np.uint8(finalImage * 255)

# finalImage = np.dstack((finalImage, alpha))

cv2.imwrite("Output/"+output_file_name+"_segmented.jpg", finalImage)

# Display the original and final images side by side
cv2.imshow("Output", np.hstack((img, finalImage)))
cv2.waitKey(0)
# Close all OpenCV windows
cv2.destroyAllWindows()

ground_truth_img = cv2.imread("Ground_Truth/"+output_file_name+".png")
gray_ground_truth_img= cv2.cvtColor(ground_truth_img, cv2.COLOR_RGB2GRAY)
gray_finalImage = cv2.cvtColor(finalImage, cv2.COLOR_RGB2GRAY)
SSIM = ssim(gray_ground_truth_img, gray_finalImage)
pixel_accuracy = pixel_accuracy(gray_finalImage, gray_ground_truth_img)
iou_score = calculate_iou(gray_finalImage, gray_ground_truth_img)
# dice_coefficient = calculate_dice_coefficient(gray_finalImage, gray_ground_truth_img)
print("SSIM score: ",SSIM)
print("Pixel Accuracy score: ",pixel_accuracy)
print("IoU score: ",iou_score)
# print("Dice Coefficient score: ",dice_coefficient)
cv2.imshow("Comparison", np.hstack((ground_truth_img, finalImage)))
cv2.waitKey(0)
cv2.destroyAllWindows()

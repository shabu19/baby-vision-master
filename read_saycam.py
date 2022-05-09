import os
import cv2
import numpy as np

def video_to_image(data_path, save_path, fps, seg_len):

    file_list = os.listdir(data_path)
    file_list.sort()

    class_counter = 0
    img_counter = 0
    file_counter = 0

    final_size = 224
    resized_minor_length = 256
    edge_filter = False
    n_imgs_per_class = seg_len * fps

    curr_dir_name = os.path.join(save_path, 'class_{:04d}'.format(class_counter))
    os.mkdir(curr_dir_name)

    for file_indx in file_list:
        file_name = os.path.join(data_path, file_indx)

        cap = cv2.VideoCapture(file_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # take every sample_rate frames (30: 1fps, 15: 2fps, 10: 3fps, 6: 5fps, 5: 6fps, 3: 10fps, 2: 15fps, 1: 30fps)
        sample_rate = frame_rate // fps + 1

        print('Total frame count: ', frame_count)
        print('Native frame rate: ', frame_rate)

        fc = 0
        ret = True

        # Resize
        new_height = frame_height * resized_minor_length // min(frame_height, frame_width)
        new_width = frame_width * resized_minor_length // min(frame_height, frame_width)

        while (fc < frame_count):

            ret, frame = cap.read()

            if fc % sample_rate == 0 and ret:

                # Resize
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                # Crop
                height, width, _ = resized_frame.shape
                startx = width // 2 - (final_size // 2)
                starty = height // 2 - (final_size // 2) - 16
                cropped_frame = resized_frame[starty:starty + final_size, startx:startx + final_size]
                assert cropped_frame.shape[0] == final_size and cropped_frame.shape[1] == final_size, \
                    (cropped_frame.shape, height, width)

                if edge_filter:
                    cropped_frame = cv2.Laplacian(cropped_frame, cv2.CV_64F, ksize=5)
                    img_min = cropped_frame.min()
                    img_max = cropped_frame.max()
                    cropped_frame = np.uint8(255 * (cropped_frame - img_min) / (img_max - img_min))

                cv2.imwrite(os.path.join(curr_dir_name, 'img_{:04d}.jpeg'.format(img_counter)), cropped_frame[::-1, ::-1, :])
                img_counter += 1

                if img_counter == n_imgs_per_class:
                    img_counter = 0
                    class_counter += 1
                    curr_dir_name = os.path.join(save_path, 'class_{:04d}'.format(class_counter))
                    os.mkdir(curr_dir_name)

            fc += 1

        cap.release()

        file_counter += 1
        print('Completed video {:4d} of {:4d}'.format(file_counter, len(file_list)))

if __name__ == '__main__':

    "------------------------------------------------------------------------------------------------------------------------"
    data_path = 'C:/Users/LENOVO/Documents/PyCharm/datasets/cat_01_pivothead'           #path to SAYCam videos
    save_path = 'C:/Users/LENOVO/Documents/PyCharm/datasets/images'                     #path to save images
    fps = 5                                                                             #sampling rate (frames per second)
    seg_len = 288                                                                       #segment length (seconds)
    "------------------------------------------------------------------------------------------------------------------------"

    video_to_image(data_path=data_path, save_path=save_path, fps=fps, seg_len=seg_len)

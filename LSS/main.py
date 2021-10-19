import cv2
import numpy as np
import utils as helper
from digit_recog import Digit_rec
from  os.path import exists
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
def load_model(file_path=""):
    if file_path != "":
        if exists(file_path):
            model = Digit_rec(1,28)
            model.load_state_dict(torch.load(file_path,map_location=torch.device("cpu")))
            return model

if __name__ == "__main__":
    image_height = 960
    image_width = 720
    cap  = cv2.VideoCapture(1)
    cap.set(3,image_width) # 3 for width
    cap.set(4,image_height) # 4 for height
    cap.set(10,150) # 10 for brigtness
    model = load_model("./digit_model.h5")
    writer = SummaryWriter(f'logs')
    seen = dict()
    while cap.isOpened():
        success,img = cap.read()
        img_corners = img.copy()
        img_result = img.copy()
        processed_img = helper.preprocess(img)
        corners = helper.find_contours(processed_img, img_corners)
        if corners:
            wraped,matrix = helper.warp_image(corners,img)
            warped_processed = helper.preprocess(wraped)
            vertical_lines, horizontal_lines = helper.get_grid_lines(warped_processed)
            mask = helper.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_processed, mask)
            squares = helper.split_into_squares(numbers)
            squares_processed = helper.clean_squares(squares)
            for i,im in enumerate(squares_processed):
                if i == 1:
                    break
                else:
                    cv2.imshow("digits-{}".format(i),squares_processed[i])
            squares_guesses = helper.recognize_digits(squares_processed, model)
            for i,temp in enumerate(squares):
                squares[i] = cv2.resize(temp, (28, 28))
            grid_ = make_grid(torch.Tensor(squares).view(81,1,28,28),normalize=True)
            writer.add_image('digit_images',grid_)
            if squares_guesses in seen and seen[squares_guesses] is False:
                continue

            # if we already solved this puzzle, just fetch the solution
            if squares_guesses in seen:
                helper.draw_digits_on_warped(wraped, seen[squares_guesses][0], squares_processed)
                img_result = helper.unwarp_image(wraped, img_result, corners, seen[squares_guesses][1])

            else:
                solved_puzzle, time = helper.solve_wrapper(squares_guesses)
                print(solved_puzzle)
                if solved_puzzle is not None:
                    helper.draw_digits_on_warped(wraped, solved_puzzle, squares_processed)
                    img_result = helper.unwarp_image(wraped, img_result, corners, time)
                    seen[squares_guesses] = [solved_puzzle, time]
                else:
                    seen[squares_guesses] = False
                if 'img_result' in [globals(),locals()]:
                    cv2.imshow('window_final', img_result)
        cv2.imshow("windo",img_corners)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

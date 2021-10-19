import cv2
import numpy as np
import operator
import torch
import time
from itertools import product
def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    
    inverted = cv2.bitwise_not(thresh, 0)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    
    result = cv2.dilate(morph, kernel, iterations=1)
    return result

def find_extreme_corners(polygon, limit_fn, compare_fn):
    
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)


def find_contours(img, original):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        top_left = find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        
        [draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []


def warp_image(corners, original):
    
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners

    
    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    
    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width)), matrix

def grid_line_helper(img, shape_location, length=10):
    clone = img.copy()
    
    row_or_col = clone.shape[shape_location]
    
    size = row_or_col // length

    
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    
    clone = cv2.erode(clone, kernel)
    clone = cv2.dilate(clone, kernel)

    return clone



def get_grid_lines(img, length=10):
    horizontal = grid_line_helper(img, 1, length)
    vertical = grid_line_helper(img, 0, length)
    return vertical, horizontal


def draw_lines(img, lines):
    clone = img.copy()
    lines = np.squeeze(lines)
    for rho, theta in lines:
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
    return clone


def create_grid_mask(vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 100)
    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask





def split_into_squares(warped_img):
    squares = []

    width = warped_img.shape[0] // 9

    
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])
    return squares


def clean_helper(img):
    
    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        return np.zeros_like(img), False
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(img), False

    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True



def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1
        else:
            cleaned_squares.append(0)

    return cleaned_squares




def recognize_digits(squares_processed, model):
    s = ""
    formatted_squares = []
    location_of_zeroes = set()

    
    blank_image = np.zeros_like(cv2.resize(squares_processed[0], (28, 28)))

    for i in range(len(squares_processed)):
        if type(squares_processed[i]) == int:
            location_of_zeroes.add(i)
            formatted_squares.append(blank_image)
        else:
            img = cv2.resize(squares_processed[i], (28, 28))
            formatted_squares.append(img)

    formatted_squares = np.array(formatted_squares)
    #print(model(torch.Tensor(formatted_squares).view(-1,1,28,28)).detach().numpy()[0])
    cv2.imshow("first_digit",formatted_squares[0])
    all_preds = list(map(int,map(np.argmax, model(torch.Tensor(formatted_squares).view(-1,1,28,28)).detach().numpy())))
    print(all_preds)
    for i in range(len(all_preds)):
        if i in location_of_zeroes:
            s += "0"
        else:
            if all_preds[i] + 1 != 9:
                s += str(all_preds[i])
            else:
                s += str(all_preds[i] + 1)

    return s



def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    width = warped_img.shape[0] // 9

    img_w_text = np.zeros_like(warped_img)

    
    index = 0
    for j in range(9):
        for i in range(9):
            if type(squares_processed[index]) == int:
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text


def unwarp_image(img_src, img_dest, pts, time):
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img






#=========================sudoku=========================

# def solve_sudoku(size, grid):
#     R, C = size
#     N = R * C
    
#     X = ([("rc", rc) for rc in product(range(N), range(N))] +
#          [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
#          [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
#          [("bn", bn) for bn in product(range(N), range(1, N + 1))])
#     print(X)
#     Y = dict()
#     for r, c, n in product(range(N), range(N), range(1, N + 1)):
#         b = (r // R) * R + (c // C)  # Box number
#         Y[(r, c, n)] = [
#             ("rc", (r, c)),
#             ("rn", (r, n)),
#             ("cn", (c, n)),
#             ("bn", (b, n))]
#     X, Y = exact_cover(X, Y)
#     for i, row in enumerate(grid):
#         for j, n in enumerate(row):
#             if n:
#                 select(X, Y, (i, j, n))
#     for solution in solve(X, Y, []):
#         for (r, c, n) in solution:
#             grid[r][c] = n
#         yield grid


# def exact_cover(X, Y):
#     X = {j: set() for j in X}
#     for i, row in Y.items():
#         for j in row:
#             X[j].add(i)
#     return X, Y


# def solve(X, Y, solution):
#     if not X:
#         yield list(solution)
#     else:
#         c = min(X, key=lambda c: len(X[c]))
#         for r in list(X[c]):
#             solution.append(r)
#             cols = select(X, Y, r)
#             for s in solve(X, Y, solution):
#                 yield s
#             deselect(X, Y, r, cols)
#             solution.pop()


# def select(X, Y, r):
#     cols = []
#     for j in Y[r]:
#         for i in X[j]:
#             for k in Y[i]:
#                 if k != j:
#                     X[k].remove(i)
#         cols.append(X.pop(j))
#     return cols


# def deselect(X, Y, r, cols):
#     for j in reversed(Y[r]):
#         X[j] = cols.pop()
#         for i in X[j]:
#             for k in Y[i]:
#                 if k != j:
#                     X[k].add(i)


def solve_wrapper(squares_num_array):
    if squares_num_array.count('0') >= 80:
        return None, None
    start = time.time()
    arr = []
    for i in squares_num_array:
        arr.append(int(i))
    print(arr)
    arr = np.array(arr, dtype=int)
    arr = np.reshape(arr, (9, 9))    
    ans = False#solve(arr)
    if not ans:
        print("solution not exist")
        return None, None
    try:
        s = ""
        for a in arr:
            s += "".join(str(x) for x in a)
        return s, "Solved in %.4fs" % (time.time() - start)
    except Exception as e:
        print("erroe occur {}".format(e))
        return None, None




# --------------------------------------new sudolu solver----------------------------
def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None
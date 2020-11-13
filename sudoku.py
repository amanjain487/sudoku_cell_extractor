import cv2
import numpy as np
import other_image_processing_functions as my_cv


# to make all pixel values non-negative
def make_absolute(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = abs(image[i, j])
    return image


# a sudoku image will have 10 horizontal/vertical lines
# so to find the longest 10 horizontal/vertical lines
# input -> image with all objects
# output -> image with largest 10 objects i.e., horizontal/vertical lines
def find_top_10_edges(source_image):
    # normalize the data to 0 - 1
    normalized = source_image.astype(np.float64) / source_image.max()
    # Now scale by 255
    normalized = 255 * normalized
    gray_image = normalized.astype(np.uint8)
    # trace all the edges
    boundary = my_cv.trace_boundary(gray_image)
    image_with_10_edges = np.zeros_like(source_image)
    # loop 10 times for 10 edges
    for i in range(10):
        # find 1st largest boundary
        path = my_cv.largest_boundary(boundary)
        # extract that object alone
        extracted = my_cv.extract_object(source_image, path)
        # remove the extracted path in all the coordinates present in the boundary
        # as all of those will have the same path
        # and also so that same path is not detected in next iteration
        for x in range(extracted.shape[0]):
            for y in range(extracted.shape[1]):
                if boundary[x][y] == path:
                    boundary[x][y] = []
        for x in range(extracted.shape[0]):
            for y in range(extracted.shape[1]):
                to_remove = False
                for k in boundary[x][y]:
                    if k in path:
                        to_remove = True
                if to_remove:
                    boundary[x][y] = []
        # add image with extracted objet to image which will contain all 10 objects after 10 iterations
        image_with_10_edges += extracted
    # return the image with 10 edges
    return image_with_10_edges


# same as above with little changes
# instead of finding 10 lines
# image contains 100 small box and as corner and then find centroid to find exact coordinate
# input -> image with 100 corners
# output -> centroids or 100 corners
def find_corners_using_centroids(source_image):
    corners = []
    # trace all the 100 boxes
    boundary = my_cv.trace_boundary(source_image)
    # iterate 100 times to extract 100 corners and find centroid of each corner box
    for i in range(100):
        # trace largest boundary
        path = my_cv.largest_boundary(boundary)
        # extract that largest corner alone
        ima = my_cv.extract_object(source_image, path)
        # remove the extracted path in all the coordinates present in the boundary
        # as all of those will have the same path
        # and also so that same path is not detected in next iteration
        for x in range(ima.shape[0]):
            for y in range(ima.shape[1]):
                if boundary[x][y] == path:
                    boundary[x][y] = []
        for x in range(ima.shape[0]):
            for y in range(ima.shape[1]):
                to_remove = False
                for k in boundary[x][y]:
                    if k in path:
                        to_remove = True
                if to_remove:
                    boundary[x][y] = []
        # find centroid of extracted object and append it to the list of corners
        current_corner = my_cv.calculate_centroid(ima)
        corners.append(current_corner)
    return corners


def verify_cell_contents(corners111, source_image, threshold=400):
    empty_cells_index = []
    non_empty_cells_index = []
    empty_cells_images = []
    non_empty_cells_images = []
    # kernel for removing noise from extracted cell
    kernel_for_removing_noise = my_cv.get_structuring_kernel((3, 3))
    # iterate through each row
    for i in range(9):
        # iterate through each cell in each row
        for j in range(9):
            # source corners for perspective transformation
            corners = np.float32([list(corners111[i, j]),
                                  list(corners111[i, j + 1]),
                                  list(corners111[i + 1, j + 1]),
                                  list(corners111[i + 1, j])])
            # destination corners for perspective transformation
            dest_corners_of_extracted_cell = np.float32([[0, 0], [50, 0], [50, 50], [0, 50]])
            # calculate perspective transformation matrix
            perspective_transformation_matrix = my_cv.get_perspective_matrix(corners, dest_corners_of_extracted_cell)
            # apply perspective transformation and extract a single cell
            extracted_cell = my_cv.bilinear_interpolate(source_image, perspective_transformation_matrix, (50, 50))
            extracted_cell = extracted_cell[1:-7, 1:-7]
            # remove unwanted black pixels
            extracted_cell = my_cv.thinning_boundary(extracted_cell, kernel_for_removing_noise, iterations=1)
            extracted_cell = my_cv.thicken_boundary(extracted_cell, kernel_for_removing_noise, iterations=1)
            cv2.imshow(str(i) + "," + str(j), extracted_cell)
            # count number of non-white pixels
            # if count greater than threshold, cell contains a number
            # else, cell is empty
            non_white_pixels_count = 0
            cell_to_append = extracted_cell.copy
            for x in range(extracted_cell.shape[0]):
                for y in range(extracted_cell.shape[1]):
                    if extracted_cell[x, y] != 255:
                        non_white_pixels_count += 1
            if non_white_pixels_count > threshold:
                non_empty_cells_index.append([i + 1, j + 1])
                non_empty_cells_images.append(cell_to_append)
            else:
                empty_cells_index.append([i + 1, j + 1])
                empty_cells_images.append(cell_to_append)
    # return index of empty and non_empty cells
    return empty_cells_images, non_empty_cells_images, empty_cells_index, non_empty_cells_index


# main function which performs and calls the necessary functions as required
def extract_cells_of_sudoku(input_parameters):
    # extract image name from image path
    image_name = input_parameters["image_path"].replace("\\", "/")
    image_name = image_name.split("/")[-1]

    # open BGR image
    # bgr_image = cv2.imread(input_parameters["image_path"])
    # open grayscale image
    grayscale_image = cv2.imread(input_parameters["image_path"], cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original_Gray_Image", grayscale_image)
    cv2.waitKey(1)

    # smoothen or blur the image
    # or reduce noise in image
    # using gaussian filter
    blurred_image = my_cv.apply_gaussian_filter(grayscale_image, my_cv.get_gauss_kernel(1))
    # convert to binary image from smoothened image
    binary_image = my_cv.convert_to_binary(blurred_image, 7, 3, 255)
    cv2.imshow("Binary Image", binary_image)
    cv2.waitKey(1)

    # normalize the image
    normalized_binary_image = my_cv.contrast_stretching(binary_image)

    # find canny edges
    canny_edges = my_cv.canny_edge(normalized_binary_image, image_name, input_parameters["sigma"],
                                   input_parameters["low_threshold"], input_parameters["high_threshold"])
    cv2.imshow("Canny Edges", canny_edges)
    cv2.waitKey(1)

    # trace all the objects
    boundaries = my_cv.trace_boundary(canny_edges)

    # find the largest box
    # the largest box in the image is assumed to be sudoku puzzle
    path_of_sudoku_puzzle = my_cv.largest_boundary(boundaries)
    # find the corners of sudoku puzzle
    corners_of_sudoku_puzzle = my_cv.find_corners(path_of_sudoku_puzzle, binary_image)
    for i in corners_of_sudoku_puzzle:
        i[0], i[1] = i[1], i[0]
    corners_of_sudoku_puzzle = np.float32(corners_of_sudoku_puzzle)
    # find perspective transformation matrix
    perspective_transformation_matrix = my_cv.get_perspective_matrix(corners_of_sudoku_puzzle,
                                                                     input_parameters["destination_corners"])
    # apply perspective transformation by bi-linear interpolation
    perspective_corrected_image = my_cv.bilinear_interpolate(binary_image, perspective_transformation_matrix,
                                                             (500, 500))
    cv2.imshow("Perspective_Corrected_Image", perspective_corrected_image)
    cv2.waitKey(1)

    # find canny edges of the sudoku again,
    # this time perspective corrected
    normalized_binary_image = my_cv.contrast_stretching(perspective_corrected_image)
    canny_edges = my_cv.canny_edge(normalized_binary_image, image_name, input_parameters["sigma"],
                                   input_parameters["low_threshold"], input_parameters["high_threshold"])

    # find the boundaries/ trace the boundaries
    boundaries = my_cv.trace_boundary(canny_edges)

    # extract the sudoku puzzle alone
    path_of_sudoku_puzzle = my_cv.largest_boundary(boundaries)
    sudoku_puzzle = my_cv.extract_object(perspective_corrected_image, path_of_sudoku_puzzle)
    cv2.imshow("Extracted Sudoku", sudoku_puzzle)
    cv2.waitKey(1)

    # find the horizontal and vertical edges
    # later find their intersection
    # which will help in finding corners
    # using which individual cells can be extracted
    normalized_binary_image = my_cv.contrast_stretching(sudoku_puzzle)
    gaussian_kernel = my_cv.get_gauss_kernel(sigma)
    sobel_kernel_x, sobel_kernel_y = my_cv.convolve_gauss_to_sobel(gaussian_kernel)
    gradient_x = my_cv.sobel_edges(normalized_binary_image, sobel_kernel_x)
    gradient_x = make_absolute(gradient_x)
    gradient_y = my_cv.sobel_edges(normalized_binary_image, sobel_kernel_y)
    gradient_y = make_absolute(gradient_y)

    # thicken the edges which will fill gaps in edges
    # thicken and thinner repeatedly until you get the edges as required
    # first find vertical edges
    # construct kernel for that
    kernel_for_edge_filling = my_cv.get_structuring_kernel((10, 2))

    # first remove noise edge pixels and then fill the edge gaps
    vertical_edges = my_cv.thinning_boundary(gradient_x, kernel_for_edge_filling, 1)
    vertical_edges = my_cv.thicken_boundary(vertical_edges, kernel_for_edge_filling, 5)
    vertical_edges = my_cv.thinning_boundary(vertical_edges, kernel_for_edge_filling, 2)
    # remove weak edge pixels
    for i in range(vertical_edges.shape[0]):
        for j in range(vertical_edges.shape[1]):
            if vertical_edges[i, j] < 1:
                vertical_edges[i, j] = 0
    # again try to fill the gaps/merge the broken edges
    vertical_edges = my_cv.thicken_boundary(vertical_edges, kernel_for_edge_filling, 2)
    vertical_edges = my_cv.thinning_boundary(vertical_edges, kernel_for_edge_filling, 1)
    # the image has all strong vertical edges
    # find largest 10 edges among them
    vertical_edges = find_top_10_edges(vertical_edges)
    cv2.imshow("Vertical Edges", vertical_edges)
    cv2.waitKey(1)

    # now to find horizontal edges
    # construct kernel for that
    kernel_for_edge_filling = my_cv.get_structuring_kernel((2, 10))

    horizontal_edges = my_cv.thinning_boundary(gradient_y, kernel_for_edge_filling, 1)
    horizontal_edges = my_cv.thicken_boundary(horizontal_edges, kernel_for_edge_filling, 4)
    horizontal_edges = my_cv.thinning_boundary(horizontal_edges, kernel_for_edge_filling, 1)
    # remove weak edge pixels
    for i in range(horizontal_edges.shape[0]):
        for j in range(horizontal_edges.shape[1]):
            if horizontal_edges[i, j] < 1:
                horizontal_edges[i, j] = 0
    # again try to fill the gaps/merge the broken edges
    horizontal_edges = my_cv.thicken_boundary(horizontal_edges, kernel_for_edge_filling, 2)
    horizontal_edges = my_cv.thinning_boundary(horizontal_edges, kernel_for_edge_filling, 1)
    # the image has all strong vertical edges
    # find largest 10 edges among them
    horizontal_edges = find_top_10_edges(horizontal_edges)
    cv2.imshow("Horizontal Edges", horizontal_edges)
    cv2.waitKey(1)

    # now find the corners using intersection of horizontal and vertical edges
    # each corner will be similar to shape of square
    # reason for that is thick edges
    corners_of_each_cell = np.zeros(horizontal_edges.shape, np.uint8)

    for i in range(horizontal_edges.shape[0]):
        for j in range(horizontal_edges.shape[1]):
            if horizontal_edges[i, j] != 0 and vertical_edges[i, j] != 0:
                corners_of_each_cell[i, j] = 255

    # fill and thicken the corners
    # there will be total of 100 corners
    # if not then tweak the input parameters, and try changing no.of iterations in each thickening and thinning process
    corners_of_each_cell = my_cv.thicken_boundary(corners_of_each_cell, my_cv.get_structuring_kernel((5, 5)),
                                                  iterations=2)
    corners_of_each_cell = my_cv.thinning_boundary(corners_of_each_cell, my_cv.get_structuring_kernel((5, 5)),
                                                   iterations=2)

    cv2.imshow("Corners", corners_of_each_cell)
    cv2.waitKey(1)

    # each corner is made up of more thN 1 PIXEL
    # find centroid of each corner
    # so that, each corner will be exactly single coordinate
    corner_coordinates = find_corners_using_centroids(corners_of_each_cell)

    # sort the corners in order
    # using which individual cell can be extracted
    corner_coordinates = np.array(corner_coordinates, dtype=np.float32)
    corner_coordinates = corner_coordinates.reshape((100, 2))
    corner_coordinates: np.array = corner_coordinates[np.argsort(corner_coordinates[:, 1])]
    corner_coordinates = np.vstack(
        [corner_coordinates[i * 10:(i + 1) * 10][np.argsort(corner_coordinates[i * 10:(i + 1) * 10, 0])] for i in
         range(10)])
    corner_coordinates = corner_coordinates.reshape((10, 10, 2))

    # extract the cells and check if they are filled or unfilled
    empty_cells_images, filled_cells_images, empty_cells_index, filled_cells_index = verify_cell_contents(
        corner_coordinates, sudoku_puzzle)
    return filled_cells_index, filled_cells_images


########################################################################################################
# ======================================================================================================
# ====================================== INPUT PARAMETERS ===============================================
# =======================================================================================================
image_path = 'E:/PythonProjects/Sudoku/Samples/s.jpg'  # <- insert image name here

# for gaussian filter
sigma = 0.5

# for edge detection
low_threshold = 0.01  # <- low_threshold value (in the range of 0 to 1)
high_threshold = 0.2  # <- high_threshold value(in the range 0 to 1)

# for adaptive thresholding
kernel_size = 7
bias = 2
changing_value = 255

# for perspective transformation of sudoku puzzle
destination_corners = np.float32([[50, 50], [450, 50], [450, 450], [50, 450]])

# for bi-linear interpolation of sudoku puzzle
destination_image_size = (500, 500)

# dict containing all the input parameters
input_params = {"image_path": image_path,
                "sigma": sigma,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "kernel_size": kernel_size,
                "bias": bias,
                "changing_value": changing_value,
                "destination_corners": destination_corners,
                "destination_image_size": destination_image_size,
                }

index_of_filled_cells, images_of_filled_cells = extract_cells_of_sudoku(input_params)

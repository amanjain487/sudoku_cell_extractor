# SUDOKU CELL EXTRACTOR

## Table of Contents
- [Description](https://github.com/amanjain487/sudoku_cell_extractor#description)
- [Requirements](https://github.com/amanjain487/sudoku_cell_extractor#requirements)
- [Installation](https://github.com/amanjain487/sudoku_cell_extractor#installation)
- [Contents](https://github.com/amanjain487/sudoku_cell_extractor#contents)
- [Execution](https://github.com/amanjain487/sudoku_cell_extractor#execution)
- [Tweaking Input Parameters](https://github.com/amanjain487/sudoku_cell_extractor#tweaking-input-parameters)
  - [Parameters for the Sample Images](https://github.com/amanjain487/sudoku_cell_extractor#parameters-for-the-sample-images)
- [Note / Warnings](https://github.com/amanjain487/sudoku_cell_extractor#note--warnings)
- [Maintainer](https://github.com/amanjain487/sudoku_cell_extractor#maintainer)


## Description
This program processes an image containing sudoku and extracts each cell and checks if the cell is filled or not. 
It gives the index of filled cells as output written in the output file. 
The images of each cell are also obtained but not saved anywhere, so once the program ends they are deleted.

## Requirements
- Python 3.8 or above
- pip 
  
## Installation
Install the following Modules.
- Numpy
  > ```pip install numpy```
- OpenCV
  > ```pip install opencv-python```

## Contents
- Outputs (10 output files)
- Samples (10 input images)
- image_process_module (module with my image processing functions)
- Observations.txt (my observations on input and output)
- README.md (readme file - the one you are reading :p)
- sudoku.py (python program to be executed)


## Execution

- Un-tar the downloaded file and open the **Sudoku** directory.
- Move your input image inside the **Samples** directory located in the **Sudoku** directory.
- Open Terminal.
- Change the path to the directory where the **sudoku.py** file is located.
- Type the following command and press Enter.
  >**```python sudoky.py```**
- Enter the input image name along with extension and press enter.
  >Eg: **```input_image.jpg```**  
-  Once executed, press any key to exit the program.
-  The output will be saved in the **Outputs** directory, with the image name as the file name of the output.

## Tweaking Input Parameters
- Problem with the binary image, try any of the following.
  - Lot of black pixels in empty cells
    ```
    bias (increase)
    ```
  - Lot of required black pixels missing
    ```
    bias (decrease)
    ```
- If the canny edges formed are not satisfactory, try changing the following parameters.
  - Missing edges
    ```
    low_threshold (decrease)
    high_threshold (decrease)
    ```
  - Extra unnecessary edges
    ```
    low_threshold (increase)
    high_threshold (increase)
    ```
- Number of corners in corners image(cannot be done by tweaking any input parameters, changes have to be done in some functions).
  - more than 100
    ```
    try thickening the edges
    ```
  - less than 100
    ```
    try thinning the edges
    ```
- Index of filled cells.
  - Any filled cell is missing
    ```
    cell_threshold (decrease)
    ```
  - Any empty cell is considered as filled cell
    ```
    cell_threshold(increase)
    ``` 
  - Both
    ```
    take some other image as input.
    ```

  ### Parameters for the Sample Images
  - sudoku_1
    - cell_threshold - 300
  - sudoku_2
    - cell_threshold - 300
  - sudoku_3
    - cell_threshold - 300
  - sudoku_4
    - cell_threshold - 400
  - sudoku_5
    - cell_threshold - 400
    - thinning the horizontal edge one extra time compared to other sample images
  - sudoku_6
    - cell_threshold - 500 (considers empty as filled)
    - cell_threshold - 525 (considers empty as filled)
    - cell_threshold - 550 (filled as empty and empty as filled)
    - cell_threshold - 600 (filled cell is considered as empty)
    
    So, this output is a failure.

  - sudoku_7
    - cell_threshold - 300
  - sudoku_8
    - cell_threshold - 400
  - sudoku_9
    - cell_threshold - 290
  - sudoku_10
    - cell_threshold - 400

## Note / Warnings
- During execution, some functions will take around 3-5 minutes based on input image size to complete execution.
- Intermediate results will be displayed during execution.
- If there is more than 7 minutes interval between 2 intermediate results, 
then please stop the execution and try tweaking the inputs or try some other image as input.
  
## Maintainer
```
Aman Kumar,
Department of Computer Science, Pune.
LinkedIn - https://www.linkedin.com/in/aman487jain/
Github - https://github.com/amanjain487
```


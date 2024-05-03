# Notes Related to this Folder

This folder contains handcrafted feature generation and classification models. 

1. Full canopy point cloud data
2. Fingerprints: Height-wise canopy point cloud data (Each canopy is segmented into 3 slices along the vertical axis)

Each point within a point cloud consists of an XYZ coordinate and RGB color values. An example vector looks like this [X, Y, Z, R, G, B]. The goal of the the scripts **fullcanopy_colorpercentages.py** and **3slice_colorpercentages.py** is to calculate the percentage of green, yellow, and brown within each point in the point cloud.

## Documentation for fullcanopy_colorpercentages.py

This Python script utilizes the Open3D library and OpenCV to perform color analysis on point cloud data stored in PCD files. The script reads point cloud data filenames and IDC (Iron Deficiency Chlorosis) scores from a CSV file, processes the corresponding point cloud files, and calculates color percentages for brown, yellow, and green in each point cloud.

### Prerequisites

- **Open3D**: Make sure you have the Open3D library installed. If not, you can install it using:
  ```bash
  pip install open3d
  ```

- **OpenCV**: Ensure you have the OpenCV library installed. If not, you can install it using:
  ```bash
  pip install opencv-python
  ```

### Usage

1. **CSV File Structure**: Ensure that the CSV file (`csv_file`) has the necessary columns, such as 'filename' and 'idc_revised', containing point cloud filenames and corresponding IDC scores.

2. **Set Paths**: Replace the placeholders in the script with the correct paths to the CSV file (`csv_file`), the folder containing point cloud data (`data_folder`), and the desired output file (`output_file`).

3. **Run the Script**: Execute the script using a Python interpreter:
   ```bash
   python script_name.py csv_file data_folder output_file
   ```
   Replace `script_name.py`, `csv_file`, `data_folder`, and `output_file` with the actual script name and paths.

### Code Explanation

- **Command-Line Arguments**:
  The script uses the `argparse` module to accept command-line arguments for the CSV file, data folder, and output file.

- **Read CSV Data**:
  ```python
  data = pd.read_csv(args.csv_file, index_col=False)
  ```
  Reads the CSV data file containing filenames and IDC scores.

- **Color Analysis Functions**:
  The script defines functions (`count_pixels_in_range` and `calculate_color_percentages`) for counting pixels within specified HSV color ranges and calculating color percentages.

- **Color Analysis Loop**:
  The main loop iterates through each point cloud file, reads the RGB values, converts them to HSV, and calculates color percentages for brown, yellow, and green.

- **Results Output**:
  The script outputs the results, including point cloud filenames, IDC scores, and color percentages for brown, yellow, and green, to a CSV file.

### Notes

- Adjust the script based on your specific CSV file structure and point cloud data.

- Ensure that the point cloud files in the data folder are in the correct format and contain RGB information.

- The color ranges for brown, yellow, and green in HSV are specified in the script. Adjust these values based on your specific color analysis requirements.



Before we get into color analysis for the 3Slice dataset, here is the documentation for how the point cloud data was sliced.


## Point Cloud Slicing and Cropping (3slice_heightwise_cropping.py)

This Python script utilizes the Open3D library to perform height-wise slicing and cropping of point clouds stored in PCD files. The script reads point cloud files from a specified folder, slices them into equal partitions along the vertical axis, and saves the resulting slices as new PCD files.

### Prerequisites

- **Open3D**: Make sure you have the Open3D library installed. If not, you can install it using:
  ```bash
  pip install open3d
  ```

### Usage

1. **Set Paths**: Replace the placeholders in the script with the correct paths to the folder containing point cloud files (`pcd_path`) and the folder where you want to store the results (`output_path`).

2. **Run the Script**: Execute the script using a Python interpreter:
   ```bash
   python script_name.py pcd_path output_path
   ```
   Replace `script_name.py`, `pcd_path`, and `output_path` with the actual script name and paths.

### Code Explanation

- **Command-Line Arguments**:
  The script uses the `argparse` module to accept command-line arguments for the folder path to PCD files (`pcd_path`) and the folder path to store the results (`output_path`).

- **Point Cloud Processing Functions**:
  The script defines functions (`points2array` and `rgb2array`) to convert XYZ and RGB points in the point cloud to NumPy arrays. It also includes a function (`pc_crop`) for cropping point clouds based on specified XYZ bounds.

- **Point Cloud Slicing Loop**:
  The main loop iterates through each PCD file, slices the point cloud into the desired number of equal partitions along the vertical axis, and saves each slice as a new PCD file.

- **Results Output**:
  The script outputs the sliced point clouds to the specified output folder.

- **Error Handling**:
  The script includes error handling to continue processing the next file if an error occurs during the slicing or cropping process.

### Notes

- Adjust the script based on your specific requirements, such as the desired number of slices and slicing orientation.

- Ensure that the PCD files in the specified folder are in the correct format and contain XYZ and RGB information.

- Check the `missing_slice` list in the script output to identify any PCD files that were not completely sliced.


## Documentation for 3Slice_colorpercentages.py

## 3-Slice Point Cloud Color Analysis

This Python script utilizes the Open3D library and OpenCV to perform color analysis on 3-Slice point cloud data stored in PCD files. The script reads point cloud data filenames and IDC (Iron Deficiency Chlorosis) scores from a CSV file, processes the corresponding 3-Slice point cloud files, and calculates color percentages for brown, yellow, and green in each point cloud.

### Prerequisites

- **Open3D**: Make sure you have the Open3D library installed. If not, you can install it using:
  ```bash
  pip install open3d
  ```

- **OpenCV**: Ensure you have the OpenCV library installed. If not, you can install it using:
  ```bash
  pip install opencv-python
  ```

### Usage

1. **CSV File Structure**: Ensure that the CSV file (`csv_file`) has the necessary columns, such as 'plotno' and 'idc_score', containing point cloud filenames and corresponding IDC scores.

2. **Set Paths**: Replace the placeholders in the script with the correct paths to the CSV file (`csv_file`), the folder containing 3-Slice point cloud data (`data_folder`), and the desired output file (`output_file`).

3. **Run the Script**: Execute the script using a Python interpreter:
   ```bash
   python script_name.py csv_file data_folder output_file
   ```
   Replace `script_name.py`, `csv_file`, `data_folder`, and `output_file` with the actual script name and paths.

### Code Explanation

- **Command-Line Arguments**:
  The script uses the `argparse` module to accept command-line arguments for the CSV file, data folder, and output file.

- **Read CSV Data**:
  ```python
  data = pd.read_csv(args.csv_file, index_col=False)
  ```
  Reads the CSV data file containing filenames and IDC scores.

- **Create a Dictionary Mapping**:
  ```python
  dictionary = dict(zip(data['plotno'], data['idc_score']))
  ```
  Creates a dictionary mapping 'plotno' to 'idc_score' for easy reference.

- **3-Slice Point Cloud Files**:
  The script searches for PCD files in the specified data folder and processes each file.

- **Color Analysis Loop**:
  The main loop iterates through each 3-Slice point cloud file, reads the RGB values, converts them to HSV, and calculates color percentages for brown, yellow, and green.

- **Results Output**:
  The script outputs the results, including 3-Slice point cloud filenames, IDC scores, and color percentages for brown, yellow, and green, to a CSV file.

### Notes

- Adjust the script based on your specific CSV file structure and 3-Slice point cloud data.

- Ensure that the 3-Slice point cloud files in the data folder are in the correct format and contain RGB information.

- The color ranges for brown, yellow, and green in HSV are specified in the script. Adjust these values based on your specific color analysis requirements.

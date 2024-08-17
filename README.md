# Football Analysis System
This project aims to develop a comprehensive football analysis system leveraging advanced computer vision and machine learning techniques. By combining object detection, tracking, image processing, and machine learning, we will extract valuable insights from football match footage.

## Key Features
-**Object Detection**: Detect players, ball, and other objects on the field.\
-**Object Tracking**: Track the movement of players and the ball throughout the match.\
-**Custom Object Detection**: Implement custom models for specific objects or actions.\
-**Team Identification**: Automatically identify and differentiate between teams.\
-**Camera Motion Analysis**: Analyze camera movements to stabilize footage and improve tracking accuracy.\
-**Player Performance Metrics**: Calculate performance metrics like speed, stamina, and more.\

## Technologies Used
-**Ultralytics YOLOv8**: State-of-the-art object detection framework.\
-**OpenCV (CV2)**: Image processing and computer vision library.\
-**K-Means Clustering**: Algorithm for player grouping and team identification.\
-**Deep Learning**: Neural networks for advanced object detection and analysis.\

## Installation

Clone the repository, create a virtual environment, activate it, and install the requirements:

  ```bash
  git clone https://github.com/khawla12-op/FootballAnalysis.git
  ```
 ```bash
 python -m venv myvenv
```
```bash
.\myvenv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
## Running the Notebook. 
1. Navigate to the `training` folder and open the Jupyter notebook.
2. Run all the cells, except the training cell.

### Needed Files

Here you will find the following files: https://drive.google.com/drive/folders/1UYkktD7iB9hwaJZBJSGs5RSLsyOFiaeT?usp=drive_link

-**`best.pt`**: This is the trained model file. Place it in the `models` folder.\
-**`testVideo.mp4`**: This is the test video output. Place it in the `output_videos` folder.
### Running the model
To run the model :
```bash
python main.py
```

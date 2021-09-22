# Hand-Gesture-Recognition-Model
This project is aimed to recognise one out of the five hand geatures. We utilize OpenCV to capture the video footage of the user. Before sending the image to the machine learning model for prediction, we apply some filters to the image. Background Removal technique is used to separate the hand from the background. Then we apply morphological transformations to make the image clear after which the image is converted to grayscale and ready to be sent to the trained machine learning model. The model consists of five CNN layer blocks followed by flattening and fully connected layers. Each CNN block consists of a CNN layer, MaxPooling Layer, Dropout layer and Batch Normalization layer. The testing accuracy achieved is around 92%.

You may find the data to be used [here](https://drive.google.com/drive/folders/1N2OArIYExcCltkeGj6DXXpU8_lEJOTz3?usp=sharing).

To run the project, follow the steps mentioned below:

1. Clone the repository.
2. pip install -r requirments.txt
3. python testing.py

On running the script, the in-built camera will be started. While capturing the background, make sure a comparatively plain surface is used to avoid the noise.

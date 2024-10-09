# COVID-19 Mask Detection Software
This repository contains the code for the **Mask Detection Software** developed to prevent the spread of COVID-19 by detecting whether a person is wearing a mask correctly. The project uses deep learning techniques to classify images into categories based on mask-wearing status.

### Authors
- **2020105721 이시온**
- **2020105723 이우일**

## Project Goal and Overview
With the ongoing global COVID-19 pandemic, one of the most effective ways to prevent the virus’s spread is by wearing a mask. This project aims to develop software that detects whether individuals are wearing masks properly. The software not only identifies if a person is wearing a mask but also checks whether they are wearing it correctly—covering both their nose and mouth—or improperly (e.g., covering only the mouth or chin).

The core goal is to assist in curbing the spread of COVID-19 by encouraging proper mask usage, using deep learning techniques to classify facial images based on mask-wearing status.

## Methods for Implementation
### Initial Approach: Using dlib
Initially, we attempted to use the `dlib` library for facial landmark detection, as it can recognize key points like the eyes, nose, mouth, and jawline. Our hypothesis was that if a person is wearing a mask, certain landmarks (like the nose and mouth) would not be detected. However, we discovered that when part of the face is obscured, `dlib` fails to detect any landmarks, making this approach unsuitable.
![Facial_Landmark](https://github.com/user-attachments/assets/e06add24-cd8e-46d2-8400-b591a22a87dd)
![Facial_Landmark_mask](https://github.com/user-attachments/assets/68ed3a27-f87d-4b39-930e-62dd080a4b30)

### Final Approach: Using TensorFlow and Keras
To overcome the limitations of the initial approach, we decided to use **TensorFlow** with the **Keras** API to train a convolutional neural network (CNN). The network is trained to classify images into the following categories:
1. Properly wearing a mask
2. Wearing a mask with the nose exposed
3. Wearing a mask with both the nose and mouth exposed
4. Wearing a mask only on the chin

The dataset used for training comes from [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net), which provides images of people wearing masks correctly and incorrectly. We extended the dataset by sourcing additional images of people not wearing masks from other GitHub repositories.

## Image Preprocessing and Data Classification
We classified the images into the following categories:
1. Mask properly worn (`N_Mask`)
2. Mask covering mouth and chin (`N_Mask_Mouth_Chin`)
3. Mask on chin only (`N_Mask_Chin`)
4. Mask not worn (`N_Mask_Nose_Mouth`)

Since the dataset from MaskedFace-Net did not include images of people without masks, we supplemented this with additional data from other sources.

## Model Architecture
The core of this mask detection software is a Convolutional Neural Network (CNN), which is highly effective in analyzing images and detecting visual patterns. In this case, it is used to identify whether a person is wearing a mask properly, partially, or not at all.

The model consists of several layers, each designed to handle specific tasks in the image analysis process:

1. Convolutional Layers: These layers are responsible for extracting important features from the input images. They use small filters to scan the image and create feature maps that highlight patterns relevant to mask detection. The model includes multiple convolutional layers:

  -   The first two layers use 32 filters of size 3x3 to detect basic patterns in the image.
  -  Deeper layers, which use 64 filters, are designed to capture more complex features.
2. Pooling Layers: After each convolutional layer, a pooling layer is applied to reduce the spatial size of the feature maps. This helps to decrease the computational load and makes the model more efficient, while retaining important information. The Max Pooling technique is used, which keeps the most important features by selecting the maximum value from a region of the feature map.

3. Dropout Layers: To prevent overfitting, where the model performs well on training data but poorly on new data, dropout layers are added. These layers randomly disable a portion of the neurons during training, ensuring that the model does not become too dependent on specific neurons and generalizes better to new images.

4. Fully Connected Layer: After the feature extraction process, the data is passed through a fully connected layer that combines all the features detected by the previous layers. This layer enables the model to make decisions about the class of the input image (e.g., whether the mask is worn correctly, partially, or not at all).

5. Output Layer: The final layer uses a softmax activation function to output probabilities for each class (proper mask use, mask covering mouth and chin only, mask on chin only). The model assigns the highest probability to the predicted class.

The model is trained using categorical cross-entropy as the loss function, which is suitable for multi-class classification tasks. The Adam optimizer is used for efficient gradient-based optimization during training. Overall, this architecture allows the model to efficiently process facial images and determine the mask-wearing status.

## Running the Software
To run the mask detection software, use the following commands:
```bash
python detect_mask.py --prototxt path_to_deploy.prototxt --model path_to_model --confidence 0.5
```
## Conclusion
This mask detection software leverages deep learning techniques to identify whether individuals are wearing masks properly. It can be used in various public settings to encourage proper mask usage and mitigate the spread of COVID-19.

# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use a simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track in the simulator without leaving the road

[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.png "Model Visualization"
[image2]: ./examples/training.png "Training Curve"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"

## Overall Project Outline

#### 1. The following are the key files

* model.py contains the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained convolution neural network 
* README.md summarizes the results
* run1.mp4 video showing vehicle negotiating the track using model.h5

#### 2. Driving the car in autonomous mode
Using the Udacity simulator and the drive.py file, the car can be driven autonomously around the track by opening the simulator and then executing the following on the command line 
```sh 
python drive.py model.h5
```

#### 3. Training  the model

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. The modified NVIDIA architecture

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 128 (model.py lines 77-88) 

The model includes RELU layers to introduce nonlinearity (code lines 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17-18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Additionally, a keras callback was used to stop training when no improvement was observed. Another callback was used to save the best (lowest validation loss) model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, data from a second track, and driving the track in the opposite direction. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to consider a convolutional neural network that could accomplish the project goals and still be trained on the available resources.

My first step was to use a convolution neural network model similar to the NVIDIA architecture ('End to End Learning for Self-Driving Cars', Bojarski et al) with slight modifications. This model seemed appropriate as it was simple and designed for the exact purposes of this project--mapping single front facing camera images to single steering commands.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used an 80/20 split. Since I was using the validation loss to stop training, I wanted to ensure I had an adequate sample size for the validation.

To combat the overfitting, I modified the model by adding a dropout layer. In addition, I used a ModelCheckpoint and EarlyStopping keras checkpoint in the fit_generator() to stop and save the model when the validation loss stopped decreasing.

The final step was to run the simulator to see how well the car was driving around track one in the Udacity simulator. There were a few spots where the vehicle fell off the track, particularly on the sharp curves, near the dirt side road, and on the bridge. To  improve the driving behavior in these cases, I added additional training data, added more recovery examples, and increased the number of filters in the final convolution layer of the neural network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, as shown in the included video.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-88) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 Normalized image                    | 
| Cropping              | 65x320x3 cropped image                        | 
| Convolution 5x5       | 2x2 stride, 24 filters                        |       
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, 35 filters                        |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, 48 filters                        |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, 64 filters                        |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, 128 filters                       |
| RELU                  |                                               |
| Dropout               | prob = 0.5                                    |
| Flattening            |                                               |
| Fully connected       | Output 100                                    |
| RELU                  |                                               |
| Fully connected       | Output 50                                     |
| RELU                  |                                               |
| Fully connected       | Output 1 (steering command)                   |

Below is a visualization of the NVIDIA architecture. The one used for this model is a slight modification.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct when it exceeded the normal lane boundaries. I recorded these recoveries at different locations on the track to capture the myriad lane boundaries. These images show what a recovery looks like starting from right:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I also recorded data on a second (more challenging) track in order to get more data points and to generalize the model.

To augment the data set, I also flipped images and angles to provide more data and remove any directional bias. For example, here is an image that has been flipped:

![alt text][image6]
![alt text][image7]

After processing the data, I had 10,019 original samples, and double that after flipping.

Using a generator() method, I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. As described above, the validation set was used to prevent model over/under fitting through the early stopping functions. The ideal number of epochs was roughly 8 as seen in the training curve shown below. I used an adam optimizer so that manually selecting the training the learning rate wasn't necessary.

![alt text][image2]



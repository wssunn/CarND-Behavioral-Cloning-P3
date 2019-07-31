# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/01_model.png "Model Visualization"
[image2]: ./examples/train_val.png
[image3]: ./examples/03_left.png "Recovery Image"
[image4]: ./examples/04_right.png "Recovery Image"
[image5]: ./examples/05_center.png "Recovery Image"
[image6]: ./examples/06_origin.png "Normal Image"
[image7]: ./examples/07_flip.png "Flipped Image"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 100 (model.py lines 83-109) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras BatchNormalization layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 92). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Here is a picture of the training and validation loss after each epoch:

![alt text][image2]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road etc.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first use convolution layer to pick up features (lanes, turns, change of road surface etc.) of image, then faltten them and use fully-connected network to find relationship between features and car steering angle.

My first step was to use a convolution neural network model similar to the AlexNet. I thought this model might be appropriate because AlexNet gradually reduce the height and width of input while increase its depth. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has droppout layer. This greatly helped.

Then I added BatchNormalization layer as well. This made the training and validation loss decreases much more quickly. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I record a few recovery movement. For example, I deliberately drive the car off track and then start recording, then I drive the car back to the track. This is very useful as it teaches the car what to do if it goes off track. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

```python
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 89745 number of data points. I then preprocessed this data by cropping 50 pixels up and 20 pixels down.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Some final thoughts
During this training process, I have found some strategies to collect data points
* Drive slowly with a lot of small adjustments rather than drive quickly with few big adjustment. This helps the neural network learn because the dataset has a much less variation. In practice, the car drives much more smoothly
* Record a lot of car turing around the bend, record the car entering and leaving the bend as well
* After a look at the steering angle, I have found it is mostly between 0.1 to 0.25. Hence the correction factor for left or right images should be around 0.5, so as to keep the car on track
* If there is going to be a big change in terms of road surface (this affects the image input), decrease the car speed
* Drive the car both clockwise and anticlockwise. Because if only drive the car in one way (say clockwise), the input will be full of steering to the right, which will make the dataset biased
* Good data is much better than a lot of data with errors!!
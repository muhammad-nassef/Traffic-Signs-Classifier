# Udacity Self-Driving Car Nanodegree Program
---
## Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_colored_images1.png "Visualization"
[image2]: ./examples/label_frequency_histogram.png "Visualization2"
[image3]: ./examples/training_gray_images.png "Gray"
[image4]: ./examples/training_colored_images2.png "colored"
[image5]: ./examples/modifiedLeNet.jpeg "model"
[image6]: ./examples/1.png "Traffic Sign 1"
[image7]: ./examples/2.png "Traffic Sign 2"
[image8]: ./examples/3.png "Traffic Sign 3"
[image9]: ./examples/4.png "Traffic Sign 3"
[image10]: ./examples/5.png "Traffic Sign 3"
[image11]: ./examples/best_guess.png "top guesses"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

##### 5 random images of the training set

![alt text][image1]


##### Label frequency histogram of the 43 classes

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I read in the paper of Sermanet and LeCun that the grayscale images gives better accuracy than the colored ones and the color doesn't add much in the model training

As a last step, I normalized the image data because it makes it easier for optimization so I normalized using this formula:
(X_train - mean(X_train)) / (max(X_train) - min(X_train))

##### Here is an example of 5 traffic sign images before and after grayscaling and normalization:

![alt text][image3]
![alt text][image4]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding,  outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding,  outputs 10x10x16  |
| RELU					| 									        	|
| Max pooling			| 2x2 stride, outputs 5x5x16    				|
| Dropout				|												|
| Convolution 5x5		| 1x1 stride, valid padding,  outputs 1x1x400	|
| RELU					|												|
| Flatten				| Flatten Last RELU output and dropout output	|
| Concatenate 			| Concatenate the two flatten outputs			|
| Dropout				|												|
| Fully connected layer	| 800 inputs , 43 outputs						|
| Output         		| the output is the fully connected layer output|

##### The model is a modified version of the one mentioned in Sermanet/Lecun Paper. I just added to dropout layers and feed the output of conv2 after max pooling not before pooling as it gave better results

 ![alt text][image5]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I've done several experiments:

* First I used the used the original LeNet model without any preprocessing and the accuracy started with around 65% and it reached to 89 % with (10 epochs , 128 batch_size, 0.001 learn_rate)
* Then I applied the preprocessing (Gray-scaling and normalization), this had a slight effect on the accuracy -> increased to around 89.5 %
* I started tuning the parameters and have done many trials on floydhub and reached to 94% when using (160 epochs, 100 batch_size, 0.0009 learn_rate)
* I converted to use cross validation by splitting the validation data from the training data and it has a direct massive effect on the accuracy for example when I use (50 epochs, 100 batch_size, 0.001 learn_rate) I get 99% validation accuracy
* Then I read the paper of Sermanet/LeCunn in which they propose a new model for traffic sign recognition in which they feed all the layers output to the classifier not only the last layer output because this provides different scales of receptive fields to the classifier as mentioned 
* I applied the new model and changed in some layers to improve the validation accuracy. After many optimizations I reached to 99 % of accuracy when using (50 epochs, 100 batch, 0.0009 learn_rate)
* Then I tried to increase the number of epochs to 100 to see the effect on accuracy and consequently I added to dropout layers to prevent over-fitting and finally I reached to 99.5% of validation accuracy (100 epoch, 100 batch_size, 0.0009 learn_rate)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 99.5 %
* test set accuracy of 95.5%


If a well known architecture was chosen:
What architecture was chosen?

* I chose the architecture proposed in Sermanet/LeCunn paper which has been used for traffic sign recognition but I modified some layers and added others


Why did you believe it would be relevant to the traffic sign application?

* The model is a modified version of the LeNet which I tried as a first step for the traffic sign application and it worked well. It also propose a better solution in which the output of each layer is fed to classifier which provide different scales to the classifier.
 
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

* I splitted the validation set out of the training set to increase the generalization not memorization of the model and it gave very high accuracy 99.5% after many optimizations and parameters tuning 
* The testing set was used only once which and it gave an accuracy of 95.5% which means it can recognize new data with 95.5% accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The images are normal but there are varieties in colors which can cause issue with the model and this point need to be tested

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h) 							| 
| Turn left ahead    	| Turn left ahead 								|
| General Caution		| General Caution								|
| Road work				| Road work										|
| Speed limit (60km/h)  | Speed limit (60km/h)							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This more than the accuracy of the test set

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit (30km/h)   						| 
| 1.     				| Turn left ahead 								|
| 1.					| General Caution								|
| 1.	      			| Road work					 					|
| 1.				    | Speed limit (60km/h)      					|


##### This image shows the certainty of model predictions, this is the best 3 predictions for each input.

![alt text][image11]




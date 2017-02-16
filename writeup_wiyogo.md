#**Traffic Sign Recognition** 
[//]: Author: Yongkie Wiyogo

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./references/traininghistogram.jpg "Training histogram"
[image2]: ./references/validationhistogram.jpg "Validation histogram"
[image3]: ./references/graycompare1.jpg "Grayscaling Comparison"
[image4]: ./references/graycompare2.jpg "Grayscaling Comparison"
[image5]: ./references/augm_rotation_minus40.jpg "Augmentation"
[image6]: ./references/new_images.jpg "New images from web"
[image7]: ./references/softmax.jpg "New images from web"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####The submission includes the project code.

Here is a link to my [project code](https://github.com/ywiyogo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook. I get three data sets:
* Training data set
* Validation data set
* Test data set

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the training set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

For my exploration of the training datasets, first I create four helper functions for the image visualization, which can be found in the third code cell. One of them can uniquely visualize the signs based on the label. It can help us to understand the datasets by seeing the images properties directly. 

Seconds, I utilize the histogram plot function from _matplotlib_ library. Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed. The x-axis represents the label IDs and the y-axis represents the number of the occurences of each ID.

![histogram of the given training datasets][image1]
![histogram of the given validation datasets][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because:
1. the color is not significant for the classification of 43 given signs. We would not use grayscale images if there were some black ring signs which has different classification
2. To improve the computational speed

There are some formulas to [convert RGB to grayscale](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale). I've tried many of them but the different grayscale calculation does not effect the accuracy. Thus, I choose the easiest one by summing all three layer and divided by three.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]

As a last step, I normalized the image data because:
1. since we deal with a lot of images, we need to have a "standard" to compare all inputs for the data consistency.
2. the CNN algorithms classify input features by multiplying the features by weights and then conducting the training (forward and backpropagation). Thus, if the input features are not normalized, then our CNN will return bad classification accuracy. [Another explanation can be see here.](http://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
There are some formulas for the normalization. I choose the on from the Lecture Introduction of TensorFlow, section 23, which is 
>(img - 128)/128

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I don't need to split the data into training and validation sets, since the given data sets has already splitted the training and the validation. 

My final training set had **42239** number of images. 
My validation set and test set had **4710** and **12630** number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because several ID label is undersampled. In order to get a better accuracy I need to generate more training input for those ID labels To add more data to the the data set, I used the following techniques:

1. Find out the undersampled ID labels
2. For each images of the undersampled ID, I generate a new image by:
    * Enlarge the Image by 1.2 (to avoid large blank spaces on the corner)
    * Rotate the image with a random degree between -40 to -15 and 15 to 40 degree. The function ignores rotation between -15 and 15 degree because the rotation is to small.
    * Translate the image to centerize the image

Here is an example of an original image and an augmented image:

![alt text][image5]

The difference between the original data set and the augmented data set is that the augmented data set is rotated randomly between -40 to -15 and 15 to 40 degree, and 1.2 times bigger than the original. 

I do not generate translated images, because translation is statistically invariance in CNN.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

|Layer|  Operations  	|     Description	        		  | Outputs shape |
|:---:|:---------------:|:-----------------------------------:|:-------------:|
| 0 | Input         	| 32x32x3 RGB image   				  | 32x32x3       |
| 1 | Convolution 5x5   | 6 filters of 5x5x1, padding: VALID  | 28x28x6       |
|   | RELU				|                                     | 28x28x6       |
|   | Max pooling	    | k: 2x2, stride: 2,                  | 14x14x6       |
|2-1| Input             | from L1 14x14x6 RGB image           |               |
|   | Convolution 5x5	| 16 filters of 5x5x6, padding: VALID | 10x10x16      |
|   | RELU          	|                                     | 10x10x16      |
|   | Max pooling   	| k: 2x2 stride: 2                    | 5x5x16        |
|   | Flatten           | .                                   | 400           |
|2-2| Input             | From L1: 14x14x6 RGB image          |               |
|   | Convolution 5x5   | 20 filters of 5x5x6, padding: VALID | 10x10x20      |
|   | RELU              |                                     | 10x10x20      |
|   | Max pooling       | k: 2x2 stride: 2                    | 5x5x20        |
|   | Flatten           | .                                   | 500           |
| 3 | Fully Connected	| Input L2-1 400 -> 100               | 100           |
| 4 | Concatenate       | Input L3 + L2-2   100+500           | 600           |
| 5 | Fully Connected   | Input L4  600 -> 100                | 100           |
| 6 | Fully Connected   | Input L5  100 -> 60                 | 60            |
| 7 | Fully Connected   | Input L6  60 -> 43                  | 43            |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

I cannot utilize GPU on my private laptop and my AWS instance cannot perform IPthon notebook very well. It hangs very often. Thus, I apply most of the standard value of LeNet:
* Batch size: 128
* type of optimizer: Adam Optimizer

After several experiments, I use 50 epochs and 0.0008 learning rate. In this folder, I recorded the epochs and the comparison in Comparison.ods file.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?



I used the itarative approach for this neural network task. Firstly I tried the LeNet from the lecture. I chose LeNet from the lecture because based on the lecture it can return more than 95% of accuracy for the handwriting dataset in grayscale format.

The problem of my initial model was that after my initial try, it has validation accuracy between 85% and 90%. This was also caused by the initial training data set.

With the final training data set, it delivered approximately between 90% to 93% validation accuracy. After, I read this recommended [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the proposed model utilized 3 times of convolution. Therefore, I tried to add the third convolution layer in LeNet. However, it did not bring a significant improvement, max only 1% accuracy.

I modified the architecture , so that on the second stage I have 2 parallel convolution layers, we can call them Layer 2-1 and Layer 2-2. The layer 3 acquires the input from L2-1 and conducts the matrix multiplication (fully connected). Layer 4 concatenates the output from Layer 3 and Layer 2-2. Afterward, I created three additional fully connected layers (Layer 5, 6, and 7). With this architecture, I can achive 95% of validation accuracy. 



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      	| Priority road    								| 
| No entry     			| No entry 										|
| General caution		| General caution								|
| Road work	      		| Road work				 	           			|
| Wild animals crossing	| Wild animals crossing      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th and 18th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image7] 
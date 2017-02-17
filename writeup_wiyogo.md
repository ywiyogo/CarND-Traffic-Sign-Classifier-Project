#**Traffic Sign Recognition** 

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

[image1]: ./references/traininghistogram.png "Training histogram"
[image2]: ./references/validationhistogram.png "Validation histogram"
[image3]: ./references/colorcompare.png "color images"
[image4]: ./references/graycompare.png "grascaled images"
[image5]: ./references/normalgraycompare.png "Normalized gray image"
[image6]: ./references/augm_rotation_minus40.png "Augmentation"
[image7]: ./references/new_images.png "New images from web"
[image8]: ./references/softmax.png "New images from web"
[image9]: ./references/LeNet2Conv-3Conv.png "LeNet 2 ConvNet vs 3 ConvNet"
[image10]: ./references/YoWiNetComparison.png "Parameter Comparison"
[image11]: ./references/YoWiNet.png "YoWiNet"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####The submission includes the project code.

Here is a link to my [project code](https://github.com/ywiyogo/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---
###Data Set Summary & Exploration

####1. Basic summary

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

####2. Include an exploratory visualization of the dataset

For my exploration of the training datasets, first I create four helper functions for the image visualization, which can be found in the third code cell. One of them can uniquely visualize the signs based on the label. It can help us to understand the datasets by seeing the images properties directly, such as dark images, very bright images and reflective images. 

Seconds, I utilize the histogram plot function from _matplotlib_ library. Here is an exploratory visualization of the data set. It is a bar chart showing how the given data set is distributed. The x-axis represents the label IDs and the y-axis represents the number of the occurences of each ID. Following, we can see the distribution of the given training and validation data sets.

![histogram of the given training datasets][image1]
![histogram of the given validation datasets][image2]

---
###Design and Test a Model Architecture

####1. Preprocessing: The submission describes the preprocessing techniques used and why these techniques were chosen

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because:

1. the color is not significant for the classification of 43 given signs. We would not use grayscale images if there were some black ring signs which has different classification
2. Through the grayscale, we can improve the contrast of an image. 
3. To improve the computational speed

There are some formulas to [convert RGB to grayscale](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale). I've tried many of them but the different grayscale calculation does not effect the accuracy. Thus, I choose the easiest one by summing all three layer and divided by three.

Here is an example of a traffic sign image before, after grayscaling, and after normalization.

![alt text][image3]
![alt text][image4]
![alt text][image5]

As a last step, I normalized the image data because:

1. since we deal with a lot of images, we need to have a "standard" to compare all inputs for the data consistency.
2. the CNN algorithms classify input features by multiplying the features by weights and then conducting the training (forward and backpropagation). Thus, if the input features are not normalized, then our CNN will return bad classification accuracy. [Another explanation can be seen here.](http://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
3. small differences of pixel values can be emphasized.

There are some formulas for the normalization. I choose the on from the Lecture Introduction of TensorFlow, section 23, which is 
>(img - 128)/128

My final training set had **42239** number of images. 
My final validation set and test set had **5010** and **12630** number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because several ID label is undersampled. In order to get a better accuracy I need to generate more training input for those ID labels To add more data to the the data set, I used the following techniques:

1. Find out the undersampled ID labels
2. For each images of the undersampled ID, I generate a new image by:
    * Enlarge the Image by 1.2 (to avoid large blank spaces on the corner)
    * Rotate the image with a random degree between -40 to -15 and 15 to 40 degree. The function ignores rotation between -15 and 15 degree because the rotation is to small.
    * Translate the image to centerize the image

Here is an example of an original image and an augmented image:

![alt text][image6]

The difference between the original data set and the augmented data set is that the augmented data set is rotated randomly between -40 to -15 and 15 to 40 degree, and 1.2 times bigger than the original. 

I do not generate translated images, because translation is statistically invariance in CNN.

####2. Model Architecture: The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

I don't need to split the data into training and validation sets, since the given data sets has already splitted the training and the validation. 

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

|Layer|  Operations     |     Description                     | Outputs shape |
|:---:|:---------------:|:-----------------------------------:|:-------------:|
| 0 | Input             | 32x32x3 Gray image                  | 32x32x1       |
| 1 | Convolution 5x5   | 6 filters of 5x5x1, padding: VALID  | 28x28x6       |
|   | RELU              |                                     | 28x28x6       |
|   | Max pooling       | k: 2x2, stride: 2,                  | 14x14x6       |
|2-1| Input             | from L1 14x14x6                     |               |
|   | Convolution 5x5   | 16 filters of 5x5x6, padding: VALID | 10x10x16      |
|   | RELU              |                                     | 10x10x16      |
|   | Max pooling       | k: 2x2 stride: 2                    | 5x5x16        |
|   | Flatten           | Flatten the 3D matrix -> 1D         | 400           |
|2-2| Input             | From L1: 14x14x6                    |               |
|   | Convolution 5x5   | 20 filters of 5x5x6, padding: VALID | 10x10x20      |
|   | RELU              |                                     | 10x10x20      |
|   | Max pooling       | k: 2x2 stride: 2                    | 5x5x20        |
|   | Flatten           | Flatten the 3D matrix -> 1D         | 500           |
| 3 | Fully Connected   | Input L2-1 400 -> 100               | 100           |
| 4 | Concatenate       | Input L3 + L2-2   100+500           | 600           |
| 5 | Fully Connected   | Input L4  600 -> 100                | 100           |
| 6 | Fully Connected   | Input L5  100 -> 60                 | 60            |
| 7 | Fully Connected   | Input L6  60 -> 43                  | 43            |


####3. Model Training: The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

I used the iterative approach for designing neural network task. Firstly I tried the LeNet from the lecture. I chose LeNet from the lecture because based on the lecture it can return more than 95% of accuracy for the handwriting dataset in grayscale image format. The code for training the model is located in the eigth cell of the ipython notebook. 

My first five experiments shows how the amount of the training data set and the convolutional layer influence the performance of the validation accuracy. The below figure shows the comparison:

![alt text][image9]

The problem of the initial model was that it returns a validation accuracy below 90%. With the generated images for training and validation data set, it deliver approximately between 90% to 92% validation accuracy. After, I read this recommended [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the proposed model utilized 3 times of convolution. Therefore, I tried to add the third convolution layer in LeNet. However, it did not bring a significant improvement, max only 1% accuracy.

Since I cannot utilize GPU (on my private laptop and IPython Notebook hangs very often in my AWS instance), I apply these standard parameters same as the LeNet model since the lecture shows how high accurate the model is:
* Batch size: 128
* type of optimizer: Adam Optimizer

My model has failed to return a good accuracy if it apply SGD optimizer. It returns an accuracy below 10%.

I was experimenting also with different values of sigma. A high value of sigma (bigger or less than 0.1) results a significant drawback. This verify the explaination from Vincent in the lecture that we should not try to start with a high probability value for our weights. The `tf.truncated_normal()` can return a range between -2sigma and 2sigma as the initial weights. Thus, we don't want to have initial weights that are higher than 0.4.

I've recorded some histories of the experiments in the file Comparison.ods.


####4. Solution Design: The project thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the problem given.

I modified the LeNet architecture, so that on the second stage, I have two parallel convolution layers, we can call them Layer 2-1 and Layer 2-2. I derived this idea from the comparison graph that three convolutional operation canprovide a good accuracy.

The layer 3 acquires the input from L2-1. It conducts a matrix multiplication to reduce the outputs(fully connected). Layer 4 concatenates the output from Layer 3 and Layer 2-2. Afterward, three additional fully connected layers are created for downsampling (Layer 5, 6, and 7). In this project, this customized model is declared as YoWiNet. 

![alt text][image11]

After my several experiments (that can be seen on the below figure), I use these parameters:
* epochs: 40
* learning rate: 0.0007.
* mu = 0
* sigma: 0.1

With this architecture, I can achive 93% of validation accuracy:
* training set accuracy of **0.951**
* validation set accuracy of **0.930**
* test set accuracy of **1.00**

![alt text][image10]

My conclusion is that, tuning the parameter and the design of CNN does not bring a very high improvement. Whatever which parameter I tuned, if othe training and validation data set not good and not big enough, the accuracy will be very low.

---
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] 

The general warning sign shall be difficult to detect since the images includes some part of other signs.

The road work sign and wild animal sign can be false classified, since it has a quite similar pattern. Furthermore, directly below the wild animal there is another small sign. This image shall give the model a challenge.

####2. Performance on New Images: The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

|Label| Image                 |     Prediction	        					| 
|:---:|:---------------------:|:-------------------------------------------:| 
|12   | Priority road      	  | Priority road    							| 
|17   | No entry     		  | No entry 									|
|18   | General caution		  | General caution								|
|25   | Road work	      	  | Road work				         			|
|31   | Wild animals crossing | Wild animals crossing           			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of **1.00**, compared to the accuracy on the test data set of **0.93**

####3. Model Certainty - Softmax Probabilities: The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

The code for making predictions on my final model is located in the 17th and 18th cell of the Ipython notebook.

For all the images, the model is 100% sure to predict the right sign (probability of 1.0).


![alt text][image8] 
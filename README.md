STRO NUMBERS
This project performs digit recognition using deep learning concepts. It can classify an image into 10 classes.We have built a Convolutional Neural Network (CNN) model using most popular Google library Tensorflow to recognize handwritten digits.
DOCS UI 

Data preprocessing steps:
 Splitting the data into training, testing and validation sets.
 Flattening the images and displaying it.
 Checking the number of instances for each digit.
 Plotting graphs and charts for easier understanding.
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.It is a dataset of 70,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.In this project we have used it to classify a given image of a handwritten digit drawn on a canvas into one of 10 classes representing integer values from 0 to 9, inclusively.
Model Construction:
Deep learning model: After manually pre-processing the dataset, we come to the part where we use concepts of Convolutional Neural Network and Transfer learning to build and train a model that classifies the handwritten digits.The 10 classes of digits are 0,1,2,3,4,5,6,7,8,9.

 This model used in Astro Numbers has been trained to detect 10 classes of objects: Numbers from 0-9 using deep learning on the CNN model. It contains 2 Convolution layers, 1 dense layer with 128 nodes (ReLU), and a softmax layer with 10 output nodes using TensorFlow and its libraries.
 The first convolutional layer is added with a small filter size (3,3) and a modest number of filters (64) followed by a max pooling layer.
 The second convolutional layer is added with a small filter size (3,3) and a modest number of filters (128) followed by a max pooling layer.



 The model is compiled using the adam optimizer and the categorical cross-entropy loss function will be optimized which is suitable for multi-class classification.We are monitoring the classification accuracy metric since we have the same number of examples in each of the 10 classes.
 The final trained model resulted in an accuracy around 99.71% on the dataset with 70,000 images.
 The model can be experimented , the user can provide the digit input to the canvas and the model will detect which number it is.
Functionalities
 Astro numbers help primary teachers to give the students a unique experience in solving maths problems.
 Works on a canvas providing an easy drawing interface.
 The web based application interface of Astro numbers uses a CNN model to classify the answers drawn by students.
 Assessment can be done by teachers and the final scores will appear on the screen.
Instructions to run:
 pip install --upgrade pandas
 pip install --upgrade matplotlib
 pip install --upgrade seaborn
 pip install --upgrade numpy
 pip install --upgrade tensorflow
 python -m http.server 8000
Project architecture:
Astro-Numbers uses Tensorflow and Keras libraries to build a sequential model with 2 Conv2D layers. We use batch_normalization at the end of every layer for higher accuracy. Activation relu worked best for the dataset. For the output layer, a dense layer was used with softmax activation.The tabular explanation of the same can be seen below.




Model loss:
The following is the validation and training loss of the above model.




As seen, there is very little noise in our model. This is due the fact that we have used the `adam` optimizer.
Model accuracy:
The following is the validation and training accuracy of the above model.




As of a typical Conv2D model, we see that the accuracy keeps improving in performance compared to the baseline.
Preview




You can check out our website in the link given here: https://data-science-community-srm.github.io/Hand-Written-Digit-Classification-Recognition/index.html
Functionalities
 Astro numbers help primary teachers to give the students a unique experience in solving maths problems.
 Works on a canvas providing an easy drawing interface.
 The web based application interface of Astro numbers uses a CNN model to classify the answers drawn by students.
 Assessment can be done by teachers and the final scores will appear on the screen.

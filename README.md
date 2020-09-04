# TensorFlow Flower Image Classifier
![alt text](https://github.com/yunglinchang/TensorFlow_image_classifier/blob/master/assets/Flowers.png?raw=true)
The project is an image classification application that identifies different species of flowers based on a deep learning model trained with TensorFlow. It is based on the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which contains 102 common flower categories in the UK. The classifier is developed into an command line app, where users can insert an image and get the flower class of the image and the probability for that class. (as shown below)
![alt text](https://github.com/yunglinchang/TensorFlow_image_classifier/blob/master/assets/inference_example.png?raw=true)

##Data Science Pipeline
The contributing model consists of the following process:
1. data visualization with Matplotlib
2. Create pipeline to preprocess image data 
3. build and train the TensorFlow classifier (attach existing model from [Feature vectors of images with MobileNet V2 trained on ImageNet](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) to current tensors)
4. save model in Keras format as "trained_model.h5"
5. load model as command line application

Code files:
* Project_Image_Classifier_Project.ipynb
* predict.py
* workspace_utils.py

## Python Packages Used
* PIL 
* logging
* Matplotlib
* NumPy
* TensorFlow
* warnings

## License
MIT License

Copyright (c) [2020] [Yung-Lin Chang]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
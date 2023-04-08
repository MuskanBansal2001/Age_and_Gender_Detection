# Age_and_Gender_Detection

Gender-and-Age-Detection
Objective :
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture.

About the Project :
In this Python Project, I have used DL to accurately identify the gender and age of a person from a single image of a face. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- 0-116.
It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

Dataset :
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

The objective of the project is to detect gender and age using facial images. Convolutional Neural Network is used to classify the images. There are 2 output types namely, gender(M or F) and age.
Download link: https://www.kaggle.com/datasets/jangedoo/utkface-new
Environment: kaggle

Additional Python Libraries Required :
OpenCV
   pip install opencv-python
Pandas
Pandas are an important library for data scientists. It is an open-source machine learning library that provides flexible high-level data structures and a variety of analysis tools. It eases data analysis, data manipulation, and cleaning of data. Pandas support operations like Sorting, Re-indexing, Iteration, Concatenation, Conversion of data, Visualizations, Aggregations, etc.

Numpy
The name “Numpy” stands for “Numerical Python”. It is the commonly used library. It is a popular machine learning library that supports large matrices and multi-dimensional data. It consists of in-built mathematical functions for easy computations. Even libraries like TensorFlow use Numpy internally to perform several operations on tensors. Array Interface is one of the key features of this library.

Matplotlib
This library is responsible for plotting numerical data. And that’s why it is used in data analysis. It is also an open-source library and plots high-defined figures like pie charts, histograms, scatterplots, graphs, etc.

Keras
Keras is a high-level library for deep learning, written in Python, that can run on different back-end engines like Theano, TensorFlow, or CNTK. It is open-source, user-friendly, and scalable for faster neural network experimentation.

Tensorflow
This library was developed by Google in collaboration with the Brain Team. It is an open-source library used for high-level computations. It is also used in machine learning and deep learning algorithms. It contains a large number of tensor operations. Researchers also use this Python library to solve complex computations in Mathematics and Physics.

scikit-learn
It is a famous Python library to work with complex data. Scikit-learn is an open-source library that supports machine learning. It supports variously supervised and unsupervised algorithms like linear regression, classification, clustering, etc. This library works in association with NumPy and SciPy.

ALGORITHM USED:

CLASSIFICATION NEURAL NETWORK
Classification is a process of finding a function which helps in dividing the dataset into classes based on different parameters. In Classification, a computer program is trained on the training dataset and based on that training, it categorizes the data into different classes.
The task of the classification algorithm is to find the mapping function to map the input(x) to the discrete output(y).
Example: The best example to understand the Classification problem is Email Spam Detection. The model is trained on the basis of millions of emails on different parameters, and whenever it receives a new email, it identifies whether the email is spam or not. If the email is spam, then it is moved to the Spam folder.

REGRESSION NEURAL NETWORK
Regression is a process of finding the correlations between dependent and independent variables. It helps in predicting the continuous variables such as prediction of Market Trends, prediction of House prices, etc.
The task of the Regression algorithm is to find the mapping function to map the input variable(x) to the continuous output variable(y).
Example: Suppose we want to do weather forecasting, so for this, we will use the Regression algorithm. In weather prediction, the model is trained on the past data, and once the training is completed, it can easily predict the weather for future days.
![image](https://user-images.githubusercontent.com/89644474/230698382-4f1cd7e7-50fc-4ee0-be7a-1e320bf26f5a.png)
![image](https://user-images.githubusercontent.com/89644474/230698455-fde2fd4b-067e-41e1-98ba-b0fbbf56f76a.png)

![image](https://user-images.githubusercontent.com/89644474/230698441-1323bcff-ec94-41ed-b3ae-2d304b3cc8da.png)
![image](https://user-images.githubusercontent.com/89644474/230698444-ab870256-0b02-4884-903f-3295dede1693.png)



# Implementing a Neural Network from Scratch

###  Introduction:

In this model we made a simple-3-layer neural network from scratch, it's an extremely valuable exercise and a essential work to designing effective models.

### Generating a dataset

We used ```sclkit-learn``` , it has some useful dataset generators and we choose the ```make_moons``` function.
The dataset we generated has two classes, plotted as red and blue points. The goal is to train a Machine Learning classifier that predicts the correct class given the x- and y- coordinates. As which the data is not linearly separable, we can't draw a straight line to separate the two classes. In fact, the hidden layer of a neural network will learn non-linear features which is one of the major advantages of Neural Network.

### Logistic Regression

We train a Logistic Regression classifier to demonstrate the point. It's input wille be the x- and y-values, then output the predicted class(0 or 1). We use the Logistic Regression class from ```sclkit-learn``` .
```
sklearn.linear_model.LogisticRegressionCV()
```
Use the ```matplotlib.pyplot``` to draw a graph shows the decision boundary learned by Logistic Regression classifier. It's a straight line so unable to capture the "moon shape" of our data.

### Training a Neural Network

Now we build a 3-layer neural network with one input layer, one hidden layer and one output layer. The input to the network will be x- and y- coordinates and its output will be two probabilities, one for class 0 and one for class 1. We need to pick an activation functions for the hidden layer. A nonlinear activation function is what allows us to fit nonlinear hypotheses and we choose ```tanh``` which performs quite well in many scenarios.

### How network makes predictions

Our network makes predictions using forward propagation, which is just a bunch of matrix multiplications and the applications of the activation function(s) we defined above. If $x$ is the 2-dimensional input to our network then we calculate our prediction $\hat{y}$ (also two-dimensional) as follows:
$$
\begin{array}\\
z_1=xW_1+b_1\\
a_1=tanh(z_1)\\
z_2=a_1W_2+b_2\\
a_2=\hat{y}=softmax(z_2)
\end{array}
$$
$z_1$ is the input of layer $i$ and $a_i$ is the output of layer $i$ after applying the activation function. $W_1,b_1,W_2,b_2$ are parameters of our network, which we need to learn from our training data. You can think of them as matrices transforming data between layers of the network. Looking at the matrix multiplications above we can figure out the dimensionality of these matrices. If we use 500 nodes for our hidden layer then $W_1\in\R^{2\times500},b_1\in\R^{500},W_2\in\R^{500\times2}$. Now you see why we have more parameters if we increase the size of the hidden layer.
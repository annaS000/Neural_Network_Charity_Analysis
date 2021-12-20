# Neural Network Charity Analysis
## Overview
> In this project, we will use the [dataset](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/resources/charity_data.csv) from Alphabet Soup’s business team, a CSV containing over 34,000 organizations funded by Alphabet Soup. Our goal is use the features of this dataset to create a binary classifier that is able to predict whether applicants will be successful if funded by Alphabet Soup. To do this we will use Python, Pandas, NumPy, Matplotlib, Sklearn, and Tensorflow to clean our data and define our model.

---

## Results
[AlphabetSoupCharity_starter_code.ipynb](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb)

### Data Preprocessing
![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/split.png?raw=true)
* What variable(s) are considered the target(s) for your model?
    >The target is the column `IS_SUCCESSFUL` since it labels whether or not the the applicant was successful and that is what we want the model to predict.
* What variable(s) are considered to be the features for your model?
    > The features are the columns of the dataframe after merging the one-hot encoded features and dropping the originals with `IS_SUCCESSFUL` removed because this is the data that will help the model determine success.

* What variable(s) are neither targets nor features, and should be removed from the input data?
    > The original columns before the encoding that we drop after merging the encoded dataframe.

### Compiling, Training, and Evaluating the Model

* How many neurons, layers, and activation functions did you select for your neural network model, and why?
    > For my original try at the model I used 1 input layer with 8 neurons, 1 hidden layer with 5 neurons, and 1 output layer with one neuron. The first 2 layers used the relu activation function and the output layer used the sigmoid function. I tried to follow an example that had been done in the module lessons to see if that would follow a similar procedure. This model was only able to achieve about 72% accuracy.

* Were you able to achieve the target model performance?

    I was unable to get over 75% accuracy but I made several attempts/ edits that I have listed below.

* What steps did you take to try and increase model performance?
    Edit 1 (1 Input Layer, 2 Hidden Layer, 1 Output Layer)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit1.png?raw=true)

    Accuracy (72.8%)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit1_acc.png?raw=true)

    Edit 2 (Number of Neurons (10,5) and 50 Epochs)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit2.png?raw=true)

    Accuracy (72.7%)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit2_acc.png?raw=true)

    Edit 3 (Number of Neurons (24,12) and 50 Epochs)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit3.png?raw=true)

    Accuracy (72.8%)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/edit3_acc.png?raw=true)

    Edit 4 (Keras Tuner)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/model_fn.png?raw=true)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/tuner.png?raw=true)
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/top_model.png?raw=true)
    Accuracy ()
    ![](https://github.com/annaS000/Neural_Network_Charity_Analysis/blob/main/images/top_model_acc.png?raw=true)
    > Tried a model function and using `keras_tuner` to give the best model and this way could not seem to get over 75% either.

---

## Summary 
Overall, is about 72% accurate in predicting the success of applicants funded by Alphabet Soup. I would say this model can still give helpful information but it may also be beneficial to try using LogisticRegression since it is also used for classification and is essentially the sigmoid activation function we use in this model.  

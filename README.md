<div align="center">

# Neural Network Fruit Classifier

A convolutional neural network trained to predict fruits. Built using PyTorch, Streamlit, and MongoDB.

![Neural Network Fruit Classifier Demo](https://github.com/brianwong778/FruitsClassifier/assets/113395187/64a11362-6635-4cd2-bc73-313be2adcc62)

</div>

## Key Features

- **Convolutional Neural Network**: Utilizes advanced CNN architectures for high accuracy in fruit classification.
- **PyTorch Integration**: Built with PyTorch to leverage its powerful and flexible deep learning capabilities.
- **Streamlit App**: Interactive web app created with Streamlit for easy demonstration of the model's capabilities.
- **MongoDB Backend**: Efficiently manages dataset and prediction results using MongoDB.

## Details

### Dataset Setup

The project utilizes the Fruits360 dataset from Kaggle. The opendatasets Python library is employed to download the dataset, which is then divided into "Test" and "Training" subdirectories. The ImageFolder class from torchvision.datasets is used to automatically assign labels based on the directory structure. In this setup, each subfolder name within the dataset directory is treated as a class label, with the images inside each subfolder representing instances of that class. The ImageFolder class also handles the initial transformation of images into tensor format, necessary for input into the neural network.

### Data Augmentation

Data augmentation techniques are applied to enhance the diversity of the training data, aiding in the reduction of overfitting and improving the model's adaptability to new information. Techniques include Random Resized Cropping, which randomly crops the image into various sizes and ratios before resizing it to its original dimensions, and Random Horizontal Flipping, which flips the image with a 50% probability. Random Rotation is also utilized, rotating the image by +/- 15 degrees, along with Color Jittering, which adjusts certain aspects of the image such as brightness, contrast, and saturation, to accommodate different lighting conditions.

### Model

The model is based on a generic base class named ImageClassificationBase, which outlines the general structure for any image classification model without being tied to a specific classifier. This class, extending torch.nn.Module, includes methods such as “training_step” and “validation_step” for processing batches of data and computing loss using cross-entropy. It also comprises functions like “validation_epoch_end” for averaging loss and accuracy across the set, and “epoch_end” for logging performance after each epoch.

The FruitsModel is a Convolutional Neural Network (CNN) built on the ImageClassificationBase. It consists of several blocks:

First Block: Begins with a convolutional layer to process the input image using 32 distinct 3x3 filters, followed by batch normalization and a ReLU activation function to introduce non-linearity. It concludes with a max pooling layer to reduce the spatial dimensions of the feature maps, helping to control overfitting.

Second Block: Features another convolutional layer increasing the filters to 64, followed by batch normalization and a ReLU activation function. It ends with a max pooling layer as well.

Third Block: Continues with a convolutional layer increasing the filters to 128, followed by batch normalization and a ReLU activation function, concluding with a max pooling layer.

The network also includes an adaptive average pooling layer to standardize the output size for the fully connected layers, and a flattening operation to convert the 2D feature maps into a 1D tensor for the dense layers. The fully connected layers include a linear transformation, a ReLU activation function, and a dropout rate to prevent overfitting, with the final layer mapping the features to the number of fruit classes in the dataset.

### Training

Device Setup: Generic device setup functions are utilized to ensure the use of a GPU, despite the redundancy given the model's training in environments like Google Colab.

DataLoader Configuration: The augmented training dataset is loaded into a DataLoader with a batch size of 128, optimizing batch processing efficiency for large datasets.

Optimizer and Learning Rate: The Adam optimizer is chosen for its adaptive learning rate capabilities, with an initial learning rate set at 0.01.

Epochs: The model undergoes 35 epochs, split into groups of 25 and 10, with checks for overfitting in between.

Training Loop: Each epoch involves a forward pass for predictions, loss calculation using cross-entropy, a backward pass for gradient computation, and optimizer steps to update model weights.

Validation Phase: At the end of each epoch, the model's performance is evaluated on a validation dataset to monitor its generalization capability.

### User Interface and Database

The trained model is saved as a .pth file and integrated into a Python application where it is redefined to use the pretrained states for predictions. MongoDB is employed to store descriptions of all 131 fruits in a straightforward format. For user interaction, Streamlit facilitates photo uploading and the prediction function, with the application retrieving the corresponding fruit description from the database upon successful prediction.

### Results

The model ended up with a 98% validation accuracy. Training and Validation loss converged around the 20’th epoch and did not diverge, indicating that there was no overfitting. 

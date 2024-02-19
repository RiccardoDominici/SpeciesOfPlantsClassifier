**Artificial Neural Networks and Deep Learning 2022/2023 - First Challenge**

1. ***Introduction***

For the year 2022, the first challenge of the course of Artificial Neural Networks and Deep Learning consisted in the creation and training of a model meant to be able to classify between 8 different species of plants, with the best accuracy possible, only using the dataset given to us by the professors for training and evaluating the performances of the model.

2. ***Data description***

In order to train our model we were given a dataset by the professors containing 3542 photos of plants, divided in 8 subfolders, one for each of the species that we needed to recognize, each one of those photos with a dimension of 96x96x3.

3. ***Setting the Model Metadata***

To create the right environment to perform the training on the models we created, we set different variables: the number of epochs, the random seed for reproducibility, the path of the dataset given by the professors, the labels dictionary necessary to associate an integer to every plant species and finally the size of the batch that we will use to fit the model. During the various training that we have done we experimented with different batch size values in order to verify if it could improve the accuracy of the model and We observed that a value lower than 64 would reduce the accuracy, while values between 64 and 512 would perform better according to the specific model in use.

4. ***Data loading, Processing, Normalization and splitting***

We defined a function named “***load\_data()”*** where we loaded the dataset with the tensorflow’s functions “***ImageDataGenerator()***” concatenated with “***flow\_from\_directory()”***, then we converted the images and labels into simple values and stored them on a numpy array to make it easier to work with. After this we splitted the data 2 times in order to obtain both a *test set* and a *validation set* so that we could later perform early stopping in the training phase to try reducing overfitting.

We tested with various different split percentages and we finally settled on a 80% train - 10% validation - 10% test split which seemed to perform the best within our tests.

We also experimented with various types of data normalization: first we simply tried to normalize the data in values between 0 and 1, then we tried to achieve a 0 average and unit variance, and finally when we stopped working with custom models but tried to use famous neural nets we used their dedicated premade data preprocessing for the corresponding model, obtaining various degrees of success and different results.

5. ***First approach: custom model***

We started to define our model by recycling the code seen during the various labs sessions, in particular we tried to recreate what was done in the third lab: an image recognition CNN.
So we proceeded by defining the function “***build\_model()”*** similarly to the lab3, which follows this structure: several convolution layers, each one with an exponential number of filters (32-64-128-...) followed by a max pooling operation with a size of (2-2), then a flattering layer and several dense layers to execute the classification, with a dropout setted to 0.3 to prevent overfitting, and finally an output layer using the softmax activation function.

![](002.jpeg)


The first change was to replace the *maxPoollingLayers* with *averagePoolingLayers,* because we thought that shades of green were important to distinguish for our

purposes, and the max pooling function could be a problem in this regard. We then proceeded to experiment with various combinations, like trying to see which number of convolution layers would produce the best result: 6 was the number that produced the best result, with an overall accuracy of 72% on the test set. Finally we managed to improve the score further by changing the kernel size of the filters, from 3x3 to 2x2, thanks to this change we obtained an accuracy of 80% on the test split.

6. ***Plotting the Test Results***

In order to better understand how the parameters were performing inside our model we decided to visualize our results thanks to “*matplotlib.pyplot”* library, in particular the correlation between our training loss and the validation loss both in the Binary Crossentropy and in the Accuracy score.

![](003.jpeg)

Furthermore, in order to understand which kind of species we were categorizing better we decided to plot, as seen during the theoretical lectures, the confusion matrix over “True Labels” and “Predicted Labels”, indicating whether a specific species was correctly categorized or miss-categorized.

7. ***Data augmentation***

Immediately we realized that the data set was too small to achieve high accuracy, so we focused on Data Augmentation techniques. First of all we perform a lot of random transformations such as shifting the image in both height and width, modifying the zoom of the picture or flipping the image both vertically and horizontally and finally modifying the brightness range of the pictures. We tested all variants of the combination and in the end we settled for these values:

- Rotation\_range = 180 
- Height\_shift\_range = 30 
- Width\_shift\_range = 30 
- Zoom\_range = 0.9 
- Horizontal\_flip = True 
- Vertical\_flip = True 
- Fill\_mode = ‘reflect’ 
- Brightness\_range = [0.2, 1.0] 

![](004.png "Original")

![](005.png "Augmented")




This set of parameters slightly improved our model, but to boost it even further we experimented over different strategies of Data Augmentation like *Cutmix*. Thanks to the *Cutmix* strategy, mixed with the first version of the data augmentation, the dataset size tripled, however, we observed that the accuracy didn’t improve as we were hoping.

8. ***Second approach: Transfer learning & Fine tuning***

Even though the custom model approach gave us good results we still decided to try to improve it by applying Transfer Learning and then Fine Tuning. Following what we have seen during the lab lecture, especially during the lab4 *“Transfer Learning and Fine Tuning”,* we imported some pre-trained models and tested them out over our FC part of the model. We started out with VGG19, trained on images from imagenet, that has a total of 21 layers made up from Convolutional Layers stacked on top of each other to increase the number of parameters so that more features could be extracted from each image, each one of those layers followed by a max pooling layer, then followed by another set of convolutional layers and so on to create a complete CNN. In particular we connected VGG19 to a GlobalAveragePooling layer followed by two Dense layers, interleaved by two Dropout layers setted to 0.3.

![](007.jpeg)

This model gave us some discrete results but not optimal, so we changed to VGG16 that has a total of 18 layers and a similar structure to VGG19 but with a decreased number of layers stacked on top of each other. This model gave us a much better result than the previous one, so we decided to proceed on using VGG16 and moved on with Fine Tuning, in the hope that it could potentially create meaningful improvements on the accuracy score, by incrementally adapting the pretrained features to new data and allowing us also to train the Fully Connected parameters. After the first training we decided to unfreeze a part of the layers so that their weights could adapt over our dataset. We experimented on different numbers of  freezed layers and we observed that we were having the best results by freezing the first 6 to 10 layers.

Thanks to Transfer Learning and Fine Tuning, combined with Data Augmentation as previously mentioned, we managed to achieve 83,7% of accuracy over the test set on our local dataset, resulting in a 77% accuracy on Codalab. Still not satisfied with our results we decided to try other pre-trained model over the ones that the keras module named ***“keras.application”*** provided, such as: Xception, ResNet50, ResNet152V2, InceptionV3, DensNet201 and EfficientNetB7. In particular for every pre-trained model we used the correct set of preprocessing for the specific model.

Despite our stubbornness in trying a lot of different models and experimenting on different trials with Fine Tuning we always ended up with lower accuracy results than the one that we managed to obtain using VGG16, so we decided to stick with it.

9. ***Third approach: Keras Tuner***

Trying to achieve better accuracy we also tried to implement a keras tuner, with this class we could tune the hyperparameters in an efficient and faster way. We wanted to keep the possibility to use Transfer Learning and Fine Tuning techniques so we kept a known bottom net, in this case VGG16 or VGG19, since it is possible to potentially tune everything, at the beginning we built an almost completely tunable model.

However it quickly turned out not to be the best strategy as it was extremely performing locally but once submitted the real accuracy dropped by almost 10%. So we first lowered the tunable parameters while fixing the other ones that we knew would lead us to higher accuracy thanks to our previous experimentations with Fine Tuning and Transfer Learning on “*VGG16*”. Doing so the accuracy over the test set grew but we still had a very high loss, so we also split the pre-augmentation dataset into another partition called “*XvalTuner*”, as large as 10% of the total dataset and we used it as a validation set only in the tuning phase, this way the difference between accuracy over the test and over the validation decreased a lot, achieve 85.2% locally and 77.6% on Codalab (final phase).

10. ***Future Improvements and conclusions:***

The final model we managed to submit during the last phases of the competition was the one mentioned above with an accuracy of 85.2% locally and even though to get to this point we spent a lot of time and energy we still think that our results can be improved, cause there is always room for improvements!

Some of the things that we thought we could test and even implement in our model are Quasi-SVM and Ensemble Techniques.

SVM stands for “*Support Vector machine*” and Quasi-SVM is a layer ( named “*RandomFuorierFeatures*” layer ) that can be stacked with a linear layer to “kernelize” linear models by applying a non-linear transformation to the Input features and then training a linear model on top of the transformed features.

While Quasi-SVM implementation is just a layer and usually doesn't modify that much the accuracy, because it focuses on training less parameters resulting in a faster training, a real improvement to our model could have been the Ensemble Technique.

Ensemble Technique (or methods) use multiple learning algorithms to obtain a better predictive performance than could be obtained from any of the consistent learning algorithms alone. There are different types of ensembles, for example stacking is one of them and involves training a learning algorithm to combine the prediction of several other learning algorithms, the easiest would be to compute the average of outputs of models in ensemble.

Our idea for this technique was to combine different models that we previously tested during Transfer Learning such as Xception, ResNet, DenseNet and our best Custom Model of 80% of accuracy. Theoretically we could stack every model that we tested and then compute the average getting both better accuracy and lower error rate than any single model.

Everything said it was a challenging yet interesting challenge that really gave us an insight about the practical world of Artificial Neural Network and that allowed us to experiment with real data on our hands and test and, more importantly, improve by a big margin our knowledge and experience on those topics.

*Riccardo Dominici Federico Filippo Diana Michele Albanese*

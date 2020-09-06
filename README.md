# SAR-image-classification-with-ML-and-DL

# ABSTRACT

This project is about the SAR(Synthetic Aperture Radar) image segmentation. This project is carried out by using Machine learning and Deep learning
technique. We have used the inception V3 model for training on the dataset. We have achieved the accuracy of 80% by using inception V3 model and 95% by using the ResNet50 model.
This code can be used for image segmentation and classification of various images which are taken from the satellites

# 1	INTRODUCTION

What is SAR imagining?

Synthetic aperture radar (SAR) imaging is a technique that allows us to remotely map the reflectivity of objects or environments with high resolution, through the emission and reception of electromagnetic (EM) signals.

Images obtained this way can be used for numerous applications, ranging from basic radar functionalities, which are the detection of objects and their geographical localization, including the estimation of some geophysical properties of complex environments, such as certain dimensions,  moisture content, roughness, density, etc. SAR also have a great day and night imaging capability.




# SAR Image Interpretation




*	SPECULAR REFLECTION:- 

In this scene, there is a river that flows in the east-west direction. As shown in the schematic above, very little energy reflects back to the radar sensor. In this case, the pixel is dark with a low db.

This can also be seen in the south-east portion with the road/airport paved surface. Again, this is a specular reflection off of a smooth surface.
 
*	DOUBLE-BOUNCE SCATTERING


On the other hand, the bright white in the centre of image can be interpreted as an urban feature. The radar is receiving double-bounce backscatter, meaning the transmitted pulses are returning back to the sensor.

It’s unclear at this scale what this object is but it’s due to double-bounce returns. Because of its values greater than -10dB, pixels will appear as a bright white.


*	DIFFUSE SCATTERING


Finally, the majority of the radar image is rough surface scattering. You have a bit of specular and double-bounce scattering.

This may be from annual cropland, vegetation or grasses or other features. It is diffuse scattering because there’s not a high or low amount of backscatter in the image.


# Scope of work

*	SAR images have wide application in remote sensing and mapping of surfaces of the Earth and other planets.

*	Applications of SAR include
o	Topography
o	Oceanography
o	Glaciology
o	Geology
o	Terrain discrimination and subsurface imaging
o	Forestry, including forest height, biomass, deforestation.


# 2.	Various model for SAR image segmentation

*	Here we are explaining three main models for SAR image segmentation
1)	Inception V3(version 3)
2)	ResNet50
3)	VGG16
 




# 	Inception V3

●	Inception-v3 is a convolutional neural network that is 48 layers deep.

●	The network has an image input size of 299 X 299 X 3.

●	The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many other.

●	As a result, the network has learned rich feature representations for a wide range of images.







# 	ResNet50

●	The ResNet-50 model consists of 5 stages each with a convolution and Identity block. Each convolution block has 3 convolution layers and each identity block also has 3 convolution layers. The ResNet-50 has over 23 million trainable parameters.
 
●	ResNet is a powerful backbone model that is used very frequently in many computer vision tasks. ResNet uses skip connection to add the output from an earlier layer to a later layer. This helps it mitigate the vanishing gradient problem.







# 	VGG 16

●	VGG takes in a 224x224 pixel RGB image

●	The VGG-16 DCNN model or the VGGNet, contains 16 convolutional layers with very small receptive fields 3 X 3, five max-pooling layers of size 2 X 2 for carrying out spatial pooling, followed by three fully-connected layers, with the final layer as the soft-max layer. Rectification nonlinearity (ReLu) activation is applied to all hidden layers.

●	The model also uses dropout regularization in the fully-connected layers.


 
2.	Model used for training dataset

*	We have used Inception V3 model For training on our dataset.

	Briefly Explained Inception V3:-


*	Inception Layer is a combination of all those layers (namely, 1×1 Convolutional layer, 3×3 Convolutional layer, 5×5 Convolutional layer) with their output filter banks concatenated into a single output vector forming the input of the next stage.

*	Along with the above-mentioned layers, there are two major add-ons in the original inception layer:

1×1 Convolutional layer before applying another layer, which is mainly used for dimensionality reduction

A parallel Max Pooling layer, which provides another option to the inception layer

*	Bigger the model, more prone it is to over fitting. This is particularly noticeable when the training data is small.



# Handling Overfitting

*	Overfitting is a trouble maker for neural networks. Designing too complex neural networks structure could cause overfitting. So, dropout is introduced to overcome the overfitting problem in neural networks.

*	The operation includes both dropping units and their connections. Dropped units can be located in both hidden layers or input / output layers. Additionally, training time reduces dramatically.




 
# 	Transfer Learning

➢		Transfer learning is a machine learning method which utilizes a pre-trained neural network. For example, the image recognition model called Inception-v3 consists of two parts:
●	Feature extraction part with a convolutional neural network.
●	Classification part with fully-connected and softmax layers.



➢	The pre-trained Inception-v3 model achieves state-of-the-art accuracy for recognizing general objects with 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher". The model extracts general features from input images in the first part and classifies them based on those features in the second part.










# 3.	Using Inception-V3 model



	Image Data Generator


➢	We have a dataset containing three classes:- Banana, Bare Soil and Surgane
➢	SAR images of L and S band.
➢	To use the Dataset as input data for the model we will use ImageDataGenertor.
 
 




	Creating Base model


➢	Import inception-v3 form keras and create the base pre-trained model.





	Load weights and compile model






# 	Train the model
 
➢	hist = model.fit_generator(generator=traindata, validation_data= testdata,validation_steps=10,epochs=10,callbacks=[lr_reduce,checkpoint],steps_ per_epoch=10)








 
# 4.	Using ResNet-50 model


	Load weights and compile mode


➢	Dataset used in this model is the same as that used in the inception model.
➢	imagenet weights are being used in this model.



	

 


# REFERENCES

➢	https://www.researchgate.net/publication/306281834_Rethinking_the_Inception_ Architecture_for_Computer_Vision
➢	Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alexander A. Alemi Google Inc. 1600 Amphitheatre Parkway Mountain View, CA Inception-v4,
Inception-ResNet and the Impact of Residual Connections on Learning.
➢	Christian Szegedy Google Inc. Wei Liu University of North Carolina, Chapel Hill Yangqing Jia Google Inc. Pierre Sermanet Google Inc. Scott Reed University of Michigan Dragomir Anguelov Google Inc. Dumitru Erhan Google Inc. Vincent Vanhoucke Google Inc. Andrew Rabinovich Google Inc.	Going deeper with convolutions
➢	SAVERS: SAR ATR with Verification Support Based on Convolutional Neural Network Hidetoshi - FURUKAWA 1778-2, Furuichiba, Saiwai-ku, Kawasaki-shi, Kanagawa, 212–0052 Japan

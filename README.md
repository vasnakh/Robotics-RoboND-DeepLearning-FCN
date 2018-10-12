[//]: # (Image References)

[image1]: ./writeup_imgs/fcn.png
[image2]: ./writeup_imgs/bad_architect_train_loss.png
[image3]: ./writeup_imgs/good2_architect_train_loss.png
[image4]: ./writeup_imgs/good_architect_train_loss.png
[image5]: ./writeup_imgs/Detected_Objects_World3.png
[image6]: ./writeup_imgs/Detected_Objects_World3_2.png
[image7]: ./writeup_imgs/Results_World1_1.png
[image8]: ./writeup_imgs/Results_World1_2.png
[image9]: ./writeup_imgs/Results_World2_1.png
[image10]: ./writeup_imgs/Results_World2_2.png
[image11]: ./writeup_imgs/Results_World3_1.png
[image12]: ./writeup_imgs/Results_World3_2.png
[image13]: ./writeup_imgs/Accuracy_SVM.png
[image14]: ./writeup_imgs/Confusion_Matrix_Unnorm.png
[image15]: ./writeup_imgs/Confusion_Matrix_Norm.png
## Deep Learning Project: Follow Me (Segmentation using Fully Connected Convolutional Networks)


---

### Writeup / README

This project implements pixel-wise segmentation to detect a target (called hero) using fully convolutional networks.

#### 1. Introduction
Given set of images and their mask (segmented images), the idea is to train a fully convolutional network to be able to segment the image pixel-wise into 3 classes: human, background, and our target human (hero). The advantage of using fully convolutional network instead of CNN is that the spacial information will not be lost which is important when the task is detection instead of classification. The image below is overall view of the fully convolutional network.

![alt text][image1]

The encoder portion is convolution network and then it reduces to deeper 1x1 convolution layer. For the decoder part bilinear sampling is used to up-sample the images and eventually get the same image size for the output which is the segmented image.

Some methods that are further used to improve the performance are:

* Instead of using regular convolution layer, we use separable convolution layer so each channel will get traversed by 1 kernel and then 1x1 convolution is used to get desired number of output channels or feature maps. Note that separable convolution is also used for decoder part. 
* Batch normalization which acts as regularization and produces better result overall. 
* Layer concatenation is used to skip connections and retain some of the finer details.


#### 2. Network architecture
Various number of layers for encoder and decoder has been used and it was found that one good architecture is the one below. More details about the other architectures will be discussed in Results section.

The architecture of the network is as follow:

* (encoder) separable convolution layer with 64 kernels of size 3x3 and stride 2 + batch normalization (layer1)
* (encoder) separable convolution layer with 128 kernels of size 3x3 and stride 2 + batch normalization (layer2)
* 1x1 convolution layer with using 256 kernels + batch normalization which results in 256 feature maps (layer3)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer 3 and layer 1) + separable convolution with 128 filters of size 3x3 and stride 1 + batch normalization (layer4)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer4 and inputs) + separable convolution with 64 filters of size 3x3 and stride 1 + batch normalization (x)

The code below shows more details for fully convolutional layer:

```
def fcn_model(inputs, num_classes):
    # Encoder: separable convolution + batch norm
    layer1 = encoder_block(inputs, 64, 2)
    layer2 = encoder_block(layer1, 128, 2)
    # 1x1 convolutiuon + batch norm 
    layer3 = conv2d_batchnorm(layer2, 256, 1, 1)
    # Decoder: bilinear up-sampling + layer concatenation
    layer4 = decoder_block(layer3, layer1, 128)
    x = decoder_block(layer4, inputs, 64)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```


#### 3. Training and Hyper-parameters
The network is trained on 4131 data points, validation set contains 1184 data points.

After trying and playing around with various values the best result was achieved by the following:


```
learning_rate = 0.0015
batch_size = 64
num_epochs = 50
steps_per_epoch = 65
```


#### 4. Results
The final score is used to check the overall performance of loss. It is calculated as below:

`average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))`

where `average_IoU` is the average of intersection over union and `n_true_positive` is number of images where the target is actually present and the network also predicts target is present, `n_false_positive` is the number of images where the target is not actually present but the network predicts it is present, and `n_false_negative` is the number of images where the target is actually present in the image but network predicts target is not present in the image.

As explained, different architectures were examined and here are some of the results of different ones:

##### 1. Network layers:

* (encoder) separable convolution layer with 32 filters of size 3x3 and stride 2 + batch normalization (layer1)
* (encoder) separable convolution layer with 64 filters of size 3x3 and stride 2 + batch normalization (layer2)
* (encoder) separable convolution layer with 128 filters of size 3x3 and stride 2 + batch normalization (layer3)
* 1x1 convolution layer with using 256 kernels + batch normalization which results in 256 feature maps (layer4)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer4 and layer2) + separable convolution with 128 filters of size 3x3 and stride 1 + batch normalization (layer5)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer5 and layer1) + separable convolution with 64 filters of size 3x3 and stride 1 + batch normalization (layer6)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer6 and inputs) + separable convolution with 32 filters of size 3x3 and stride 1 + batch normalization (x)

Below is the train and validation loss graph:

![alt text][image2]

Even though the validation loss was `~0.0300` and train loss was `~0.016` the final score was not very good, here are the details:

```text
## performance on images while the quad is following behind the target
number of validation samples intersection over the union evaluated on 542
average intersection over union for background is 0.9947026676000658
average intersection over union for other people is 0.3223483185239623
average intersection over union for the hero is 0.8948587219607312
number true positives: 539, number false positives: 0, number false negatives: 0
```


```text
## performance on images while the quad is on patrol and the target is not visable
number of validation samples intersection over the union evaluated on 270
average intersection over union for background is 0.9836399369844645
average intersection over union for other people is 0.6626240981048774
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 94, number false negatives: 0
```

```text
## how well the neural network can detect the target from far away
number of validation samples intersection over the union evaluated on 322
average intersection over union for background is 0.9954423572649268
average intersection over union for other people is 0.40262367903378987
average intersection over union for the hero is 0.16530204882597546
number true positives: 112, number false positives: 2, number false negatives: 189
```

```text
### Final results
true_pos/(true_pos+false_neg+false_pos) = 0.6940298507462687
average_IoU = 0.5301533035101009
final_score = 0.3679422181077566
```


It seems that this architecture is not generalizing well since there are too many parameters (overfitting) or it might be just the fact of lacking enough data to train this network well enough.


To further investigate another architecture similar to network 1 was used which is network 2.

##### 2. Network layers:

* (encoder) separable convolution layer with 32 filters of size 3x3 and stride 2 + batch normalization (layer1)
* (encoder) separable convolution layer with 64 filters of size 3x3 and stride 2 + batch normalization (layer2)
* (encoder) separable convolution layer with 128 filters of size 3x3 and stride 2 + batch normalization (layer3)
* 1x1 convolution layer with using 256 kernels + batch normalization which results in 256 feature maps (layer4)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer4 and layer2) + separable convolution with 128 filters of size 3x3 and stride 1 + batch normalization (layer5)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer5 and layer1) + 2 X separable convolution with 64 filters of size 3x3 and stride 1 + batch normalization (layer6)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer6 and inputs) + 2 X separable convolution with 32 filters of size 3x3 and stride 1 + batch normalization (x)

Below is the train and validation loss graph for this network:

![alt text][image3]

The validation loss was `~0.0280` this time and train loss was `~0.0141` for the last epoch.

The details and final score are as below:

```text
## performance on images while the quad is following behind the target
number of validation samples intersection over the union evaluated on 542
average intersection over union for background is 0.9954920909379806
average intersection over union for other people is 0.3460475583480535
average intersection over union for the hero is 0.9127093486293335
number true positives: 539, number false positives: 0, number false negatives: 0
```


```text
## performance on images while the quad is on patrol and the target is not visable
number of validation samples intersection over the union evaluated on 270
average intersection over union for background is 0.9873359209874307
average intersection over union for other people is 0.7338875224785798
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 77, number false negatives: 0
```

```text
## how well the neural network can detect the target from far away
number of validation samples intersection over the union evaluated on 322
average intersection over union for background is 0.9963769970768773
average intersection over union for other people is 0.43122490129954105
average intersection over union for the hero is 0.22401779779096004
number true positives: 124, number false positives: 4, number false negatives: 177
```

```text
### Final results
true_pos/(true_pos+false_neg+false_pos) = 0.7198697068403909
average_IoU = 0.5683635732101467
final_score = 0.40914771882554535
```


It was observed that this network is performing pretty good so the conclusion is that neither of the above assumptions were correct; simply network 1 was not really good enough to perform well. 

##### 3. Network layers:

* (encoder) separable convolution layer with 64 kernels of size 3x3 and stride 2 + batch normalization (layer1)
* (encoder) separable convolution layer with 128 kernels of size 3x3 and stride 2 + batch normalization (layer2)
* 1x1 convolution layer with using 256 kernels + batch normalization which results in 256 feature maps (layer3)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer 3 and layer 1) + separable convolution with 128 filters of size 3x3 and stride 1 + batch normalization (layer4)
* (decoder) bilinear up-sampling by factor of 2 + layer concatenation (layer4 and inputs) + separable convolution with 64 filters of size 3x3 and stride 1 + batch normalization (x)



![alt text][image4]

 `~0.0300` and train loss was `~0.016` the
 
 The validation loss was `~0.049` this time and train loss was `~0.0218` for the last epoch.
 
 The details and final score are as below:
 
```text
## performance on images while the quad is following behind the target
number of validation samples intersection over the union evaluated on 542
average intersection over union for background is 0.9921249449599926
average intersection over union for other people is 0.30181393190628575
average intersection over union for the hero is 0.8883976764825544
number true positives: 539, number false positives: 0, number false negatives: 0
```


```text
## performance on images while the quad is on patrol and the target is not visable
number of validation samples intersection over the union evaluated on 270
average intersection over union for background is 0.9828554390939547
average intersection over union for other people is 0.6968990662683849
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 93, number false negatives: 0
```

```text
## how well the neural network can detect the target from far away
number of validation samples intersection over the union evaluated on 322
average intersection over union for background is 0.991947216050213
average intersection over union for other people is 0.35231146799815016
average intersection over union for the hero is 0.20993209433105278
number true positives: 155, number false positives: 3, number false negatives: 146
```

```text
### Final results
true_pos/(true_pos+false_neg+false_pos) = 0.7414529914529915
average_IoU = 0.5491648854068036
final_score = 0.40717994708581384
```

Network 3 was chosen to be a good network since it has simpler model and performs faster and probably better when there is not enough data. The code that ran with Netowrk 2 is located in `code/More_Complex_Architecture` folder and Network 3 in `code` folder.

To see images of the segmentated image by the network and the true segmented image side by side please refer to ipython notebook (located in `code` folder)

Note: Also different learning rates was used starting from 0.2 and it was observed that 0.0015 was the best choice to not have so much ossilation and stable result.
#### 5. Further improvements:
To further improve the performace, using more data would be helpful: see where the weaknesses of the network are and gather data related to those areas. Also using different optimization methods might help. More compelex architecture with more layers can also improve performance if more data is gathered.
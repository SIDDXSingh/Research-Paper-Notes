# Notes [Spatial Pyramid Pooling in Deep  Convolutional Networks for Visual Recognition ](https://arxiv.org/abs/1406.4729)

###### tags: `notes` `object detection` `sppnet` `convolutional neural networks`


## Brief Outline

The paper proposes a workaround to feeding a fixed size input to CNNs. Resizing the input can lead to reduction in recognition accuracy for images/sub - images. Cropping and resizing often result in unwanted geometric distortions.

The authors add a 'spatial pyramid pooling' layer after convolution layers to remove the fixed size constraint to network.

1. SPP is an extension to Bag - of - Words (BoW) model. It partitions image into divisions from finer to coarser levels and aggregates local features in them.
2. The network can take variable sized images, therefore, imparting scale invariance and reduces the chance for overfitting.
3. There is a speed up of over 100 times on R-CNN.
4. Region proposals are extracted using EdgeBoxes.

## Deep Networks with Spatial Pyramid Pooling

1. In general, the architectures defined have a set of convolutional layers, followed by pooling layers. The last layers are often fully connected, with $N$ way softmax as output ($N$ is total categories). Such networks require fixed size.
2. FC layers are the reason fixed size inputs are required.

### SPP Layer

1. To generate fixed sized vector invariant to input size, BoW can be used. It pools the features together.
2. The responses of each filter is max pooled.
3. The feature maps from last convolutional layer of CNN are passed through the SPP layer. With total number of bins fixed to $M$ and number of filters in last layer as $k$, a $kM$ dimensional vector is obtained.
4. The obtained vector is the input to FC layers.
5. Now that the network is invariant to size, size augmentation techniques can be used to achieve scale invariance (like in SIFT vectors).
6. One pyramid level has the entire image. It is required to reduce model size and reduce overfitting.

## Training the Network

### Single Size Training

1. A fixed size input is taken ($224 \times 224$). Cropping is done for data augmentation.
2. This is to enabe multilevel pooling behaviour. There is a gain in accuracy with this step.

### Multisize Training

1. For experimentation, two fixed sizes are used, $180 \times 180$ and $224 \times 224$. Resizing of images are done. With SPP layer, the vector obtained with both images is the same.
2. The networks obtained through these two images share same parameters. Therefore, two same networks are obtained with different sized inputs.
3. Each epoch is utilized to train one network. Then the next epoch is utilized to train the other network.

NOTE: Here only two sizes are used for training. Testing can be done on any sized input image.

## SPP-Net for Image Classification
### Experiments on ImageNet 2012 Classification

1. Network trained on 1000-category training
    set of ImageNet 2012.
2. Images is resized so that small dimension is 256.
3. Centre Crop of size $224\times224$  is taken.
4. Dropout is used on the two fully-connected layers.
5. The learning rate starts from 0.01, and is divided by 10
    (twice) when the error plateaus.
#### SPP on different network architechtures:
- ZF5, ConvNet-5 and Overfeat 5 and 7 have been used
- In the baseline models, the pooling layer after the last convolutional layer generates $6\times 6$ feature maps,     with two 4096-d fc layers and a 1000-way softmax layer        following.

#### Multi Level Pooling
- It imporves accuracy.
- The training and testing sizes are both
    $224\times 224$
- The last convolutional layer is follwed by the SPP layer in the above CONV-Nets. 4 level pyramid is used {$6\times 6$, $3\times 3$, $2\times 2$, $1\times 1$}. This improves accuracy from the one used in without SPP layer.
- Gain of multi-level pooling is because the multi-level pooling is robust to the variance in object deformations and spatial layout.

#### Multi-size Training Improves Accuracy:
- Training size: $224 ,180$
    Test Size: $180$
- The top-1/top-5 errors of all architectures further drops.
- Using the two discrete sizes of 180 and 224, evaluated using a random size uniformly sampled from $[180,224]$, The top-1/5 error of SPP-net (Overfeat-7) is $30.06$%/$10.96$%, which is worse than 2 size training(As 224 size is less visited) but is still better than single size.



![](https://i.imgur.com/ImQ947V.png)



#### Full-image Representations Improve Accuracy:

- Image is resized keeping aspect ratio intact and minimum width as $256$.
- The top-1 error rates are all reduced by the full-view representation showing the importance of maintaining the complete content.
- Combination of multiple views is substantially better than the single full-image view.

#### Multi-view Testing on Feature Maps:
- Resize an image so min(w; h) = s where s represents a predefined scale(like $256$), in test phase.
- Compute the convolutional feature maps from the entire image.
- Any window in an image is mapped to the feature maps and then use SPP to pool the features from this window.
- The pooled features are then fed into the fc layers to compute the softmax score(Which are averaged for final predictions) of this window.
- For the standard 10-view: $s=256$, and the views are $224 \times 224$ windows on the corners or center.
- Further this method is applied to extract multiple views from multiple scales:
    1. Resize the image to six scales **s ∈ {224; 256; 300; 360; 448; 560}**
    2. Compute the feature maps on the entire image for each scale.
    3. Use $224 \times 224$ as the view size for any scale, so these views have different relative sizes on the original image for different scales.
    4.  We use 18 views for each scale: one at the center, four at the corners, and four on the middle of each side, with/without flipping (when s = 224 there are 6 different views).
    
The combination of these 96 views reduces the top-5 error from $10.95$% to $9.36%$%. Combining the two full image views (with flipping) further reduces the top-5 error to $9.14$%.

### Experiments on VOC 2007 Classification:
- Above mentioned models are first trained on ImageNet Dataset.
- These representations are extracted from the images in the target datasets and SVM classifiers are retrained.
- Data Augmentation not used while training SVM.
- L2 normalization is used.
- Images are resized so that the smaller dimension is s and use the same network to extract features.
-  $s=392$ gives the best results
-  Overfeat-7, multi-size trained gives the best results.
### Experiments on Caltech 101:
- $30$ images are randomly sampled per category for training and up to $50$ images per category for testing.
- The scale $224$ has the best performance among the scales  tested on this dataset.
- Image warping is also tested but it reduces the accuracy due to distortions.

![](https://i.imgur.com/AITkMAy.png)
![](https://i.imgur.com/1kRHcLE.png/ )



## SPP-Net for Object Detection

* The paper states the pros and cons of the RCNN network and mentions that the major bottle neck is the excessive amounts of regional proposals.
* The SPP Net can be used for object detection, the feature maps are generated from each views and are then hierarchically pooled as mentioned in the above sections.
* The SPP Net extracts window-wise features from regions of the feature maps, while R-CNN extracts directly from image regions.
* The Overfeat detection method also extracts features using CNNs, but needs predefined window size.

### Detection Algorithm
* Initially the fast mode of the selective search algorithm is utilized to generate region proposal which are then resized according to the requirement.
* The next step is to extract feature maps using the SPP-net model of ZF-5 using 4 level spatial pyramid to generate 12,800-d $256\times50$ representation for each window.
* These representations are passed through fc layer and the subsequent vectors are classified using binary SVMs.
* The SVM training is done according to the steps mentioned in the RCNN paper.
* This method is improved using multi scale feature extraction. The image is resized such that **min(w, h) = s ∈ S = {480, 576, 688, 864, 1200}**, and compute the feature maps of conv5 for each scale.
* The majorly used strategy is to pool channel wise.
* Another well followed technique is, for each candidate window, a single scale s ∈ S is chosen  such that the scaled candidate window has a number of pixels closest to $224\times224$.
* These windows are used to extract feature maps.
* If the pre-defined scales are dense enough and the window is approximately square, this method is roughly equivalent to resizing the window to $224\times224$ and then extracting features from it.
* The fc layers are fine tuned according to the the steps given in RCNN.
* It is also to be noted that during fine-tuning, the positive samples are those overlapping with a ground-truth window by $[0.5, 1]$, and the negative samples upto $0.5$ (not including it).
* The bounding box regression technique is also adopted from the RCNN paper.
* The features used for regression are the pooled features from conv5 (as a counterpart of the pool5 features used in RCNN).

### Detection Results
* The detection task is carried out on the Pascal VOC 2007 dataset.
* The below table shows the result of different layers using 1 and 5 scales.
![](https://i.imgur.com/MDugiBN.png)

### Complexity and Running Time
* Even though the SPP Net has a comparable accuracy compared to the RCNN, the SPP Net is much faster.
* The complexity of the convolutional feature computation in R-CNN is $O(n ·227^2)$ with the window number $n (∼2000)$. This complexity of our method is $O(r · s^2)$ at a scale s, where r is the aspect ratio.
* The below table gives an idea about how fast the SPP Net is in handling 1 image compare to the RCNN.
![](https://i.imgur.com/DANz2PY.png)
* For obtaining the bounding box, the authors use the [Edge Box method](https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf) instead of the selective search algorithm as it is much more faster in comparison.
* And this edge box method for giving regional proposal was responsible for making the entire pipeline fit for real time usage.

### Model combination for Detection
* The authors train another Alexnet with random initializations with the intention to combine two models to boost the results.
* Given the two models, the authors first use either model to score all candidate windows on the test image.
* Then they perform non-maximum suppression on the union of the two sets of candidate windows (with their scores).
* A more confident window given by one method can suppress those less confident given by the other method. After combination, the mAP is boosted to $60.9$%.
* In $17$ out of all $20$ categories the combination performs better than either individual model.

## Conclusion
* The major advantages of SPP is it handles different scales, aspect ratios and sizes of the images.
* Deep learning networks coupled with the SPP shows remarkable increase in the performance of the tasks involving object detection, classification and localization.
* The boost in performance is both in accuracy and the time complexity.

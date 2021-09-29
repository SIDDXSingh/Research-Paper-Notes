# Notes [Fast R - CNN](https://ieeexplore.ieee.org/document/7410526/)

###### tags: `notes` `object detection` `rcnn`

##### Authors: Ross Girshick

##### Notes written by Arihant Gaur, Saketh Bachu and Siddharth Singh

## Brief Outline

The paper is an extension of R - CNN, with following achievements over previous work:
1. 9 times faster training on deep VGG16 network than R - CNN and 3 times faster than SPPNet.
2. 213 times faster test time than R - CNN and 10 times faster than SPPNet.

## Introduction

1. R - CNN is known to be very slow as the ConvNet is traversed for every region proposal. Detection takes as long as 47 seconds per image on a GPU.
2. However, the drawback common to both R - CNN and SPPNet is that they are multi - stage pipelines. These include fine - tuning a network with log loss, training SVMs and fitting bounding - box regressors.
3. Fast R - CNN overcomes these difficulties with the following contributions:
* High mAP than R - CNN and SPPNet
* Single stage training
* In training stage, backpropagation will take place across all network layers, rather than in stages.
* No disk storage required for feature caching.

## Fast R - CNN architecture and training


![](https://i.imgur.com/Bdaz8tq.png)
`Image Source`:[Cogneethi(Youtube)](https://www.youtube.com/watch?v=ZBPQ7Hd46m4)

* The method of sampling in R - CNN and SPPNet is you take and image and extract some fixed number of ROIs, and then these ROIs are batched and are provided to the neural network. The elementary difference in R - CNN is a multi - staged sampling. First we batch the images (N) and then extract a prefixed number of ROIs (R/N) from each image. Since these ROIs are inherently from same image they share computation and memory which speeds up the entire process since we do not need to repeat the calculations.
* A Fast R-CNN network takes as input an entire image and a set of object proposals.
* The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map.
* For each object proposal **ROI Poling Layer** feature vector of fixed from feature map and is then fed into a FC-Layer which further truncates into a softmax layer to predict the object class and a bounding box regressor.


### <u> ROI Pooling Layer </u>:
1. Each ROI(Region proposal) is defined by four coordinates (r,c,h,w) where (r,c) is centre of ROI box from top left and (h,w) are height and width respectively.
2. Each ROI is divided into a spatial grids of $H\times W$.(H,W are hyperparameters. In this paper it is taken as $7\times 7$ with VGG-16).
3. This can be special case of SPP pooling but with only one pyramid level.

### <u> Initialization of Network </u>
* In this paper three Networks were used namely: AlexNet, VGG-16 and VGG_CNN_M_1024. 
* Any network for Fast-RCNN has to undergo following transformations:
    1. The last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g.,H = W = 7 for VGG16).
    2. Second, the network’s last fully connected layer and softmax(which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K+1 categories and category-specific bounding-box regressors).
    3. Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images.

### <u> Fine Tuning for detection:</u>
*  Fast RCNN training, stochastic gradient descent (SGD) minibatches are sampled hierarchically, first by sampling N images and then by sampling $\frac{R}{N}$ RoIs from each image.
*  Along with this FAST-RCNN also uses streamlined training process with one fine-tuning stage that jointly optimizes a softmax classifier and bounding-box regressors, rather than training a softmax classifier, SVMs, and regressors in three separate stages.

#### 1. Multi-task loss:
* FAST-R has 2 output layers:
    * Probability of K+1 classes using the softmax layer.
    * Bounding Box regressor which returns t^k^ =(t^k^~x~,t^k^~y~,t^k^~w~,t^k^~h~), for each of the K object classes, indexed by k, t^k^ specifies a scale-invariant translation and log-space height/width shift relative to an object proposal (As given in [RCNNpaper](https://ieeexplore.ieee.org/document/7112511)).
    * A multitask loss L is used as given by:
        **L(p, u, t^k^, v) = L~cls~(p, u)+ $\lambda$[u $\ge$ 1]L~loc~(t^u^, v)**
        where,
        u: ground truth class,
        v: ground-truth bounding-box regression target,
        L~cls~(p, u)= -log p~u~ is logg loss for u,
            
        L~loc~:It is defined over a tuple of true bounding-box regression targets for class u, v =(v~x~, v~y~, v~w~, v~h~), and a predicted tuple tu =(t^k^~x~,t^k^~y~,t^k^~w~,t^k^~h~) ,again for class u. The Iverson bracket indicator function[u $\ge$ 1] evaluates to 1 when u $\ge$ 1 and 0 otherwise. u=0 means background class and L~loc~ is ignored for that.
        
    * For bounding box regression:
    
        L~loc~(t^u^, v)=$\sum_{i \epsilon \{x,y,w,h\}} {smooth_{L1}}(t^u_{i}, v_i )$
        where:
            ![](https://i.imgur.com/N1unbWU.png)
    * $\lambda$ has been taken as 1 in the paper. It controls the balance between the two task losses.
    * v~i~ can be normalized for zero mean and unit variance. 


#### 2. Mini batch Sampling:
* Each SGD minibatch is contructed using N=2 images.
* Mini-batches of size R = 128, sampling 64 RoIs from each image.
* 25% of the RoIs from object proposals is taken that have intersection over union (IoU) overlap with a groundtruth bounding box of at least 0.5.
* These ROIs have u $\geq$ 1.
* During training, images are horizontally flipped with probability 0.5.
#### 3. Back-propagation through RoI pooling layers:
* Backpropagation routes derivatives through the RoI pooling layer.
* Let x~i~ $\epsilon$ $\mathbb{R}$  be the i-th activation input into the RoI pooling layer and let y~rj~ be the layer’s j-th output from the r^th^ RoI.
* R(r, j) is the index set of inputs in the sub-window over which the output unit y~rj~ max pools. A single x~i~ may be assigned to several different outputs y~rj~ .
* The RoI pooling layer’s backwards function computes partial derivative of the loss function with respect to each input variable xi by following the argmax switches:
![](https://i.imgur.com/MOBi8Oa.png)



#### 4. SGD hyper-parameters:
* FC layers for softmax and bounding box regression initilized from zero-mean Gaussian distributions with standard deviations **0.01 and 0.001**, respectively.
* Biases =0 
* Learning Rates: 
    * lr=1 for weights,
    * lr=2 for biases,
    * global lr=0.001
* For Pascal VOC-2007 and 2012:
    * SGD for 30k mini-batch iterations 
    * then lower the learningrate to 0.0001 and train for another 10k iterations.
* Momentum=0.09
* Parameter decay of 0.0005

#### 5. Scale Invariance:
* Two ways:
    1. Brute force learning
    2. Using image pyramids
## Fast R - CNN Detection

1. During detection, the whole image is given to ConvNet, along with $R$ proposals ($R = 2000$)
2. For each test RoI, forward pass outputs posterior probability distrivution and set of predicted bounding - box offsets.
3. Assign detection confidence to the RoI test. Perform independent NMS for each class.

### Truncated SVD for faster detection

1. Since there are a lot of RoIs that need to be processed, the detection stage takes a lot of time.
2. For this, the FC layers are compressed with truncated SVD. The weight matrix $W$ can be factorized as $W \approx U_{u\times t} \Sigma_t V^T_{v \times t}$. In truncated SVD, parameter count is reduced from $uv$ to $t(u+v)$.
3. A single FC layer is divided into two FC layers (no nonlinearity in between). First layer has weights $\Sigma_t V^T$ and second has $U$ as weights.
4. Experimentally, it reduced detection time by more than $30\%$ with $0.3mAP$ tradeoff.

## Main Results

### Experimental Setup
Three types of networks are utilised:
1. Imagenet Pre - trained Alexnet referred as a small network.
2. Imagenet Pre - trained VGG CNN M 1024 referred as a medium network.
3. Imagenet Pre - trained very deep VGG16 referred as a large network.
5. This paper achieves SOTA mAP on VOC07, 2010 and 2012and also is faster compared to RCNN and SPPNet.
6. The following tables display the results on VOC 2007, VOC 2010, VOC 2012 datasets. 
![](https://i.imgur.com/ILCDHF2.png)

## Fine - tuning convolution layers

1. As per one of the ablation that was reported in the paper, fine tuning the convolutional layers increased the mAP more when compared to fine tuning only the fc layers.
2. Further it was reported in the paper that fine tuning all the convolutional layers was not necessary and in the case of VGG16, they found that, it was only necessary to update layers from conv3_1 and up (9 of the 13 conv layers).

## Design Evaluation

### Utility of multi - task training loss
1. It avoids stage wise or sequential training.
2. In the case of Fast RCNN, the multitask training was performing much better compared to all the other training norms.
3. The mAP for the different norms are as follows:
   Only $L_{cls}$ < Disabled bbox at test time < Stage wise < Multitask training
   
### Scale Invariance
1. Testing the Fast RCNN network with 5 different scales increased the mAP only by a small amount.
2. It is also reported that deep Conv Nets are inherently skilled at learning scale invariance.
3. Single scale processing offers the best trade-off between speed and accuracy, especially for deep models.

### Effect of increasing the number of training proposals
1. While using the selective search's quality mode, the mAP increases slightly initially and start to fall as the number of proposals start to rise.
2. While measuring the effect of increasing the region proposals on the Average Recall, it is observed that the AR increases.
3. Therefore, the AR metric should not be correlated with the mAP and it should be used with care.
4. It is also reported that while adding more dense boxes, mAP falls more strongly than when adding more selective search boxes.
5. Using dense boxes with SVMs performs the worst.

### Conclusion
1. Effect of sparse and dense proposals was easier to probe with Fast RCNN.
2. If certain methods that utilize dense proposals are introduced, they may accelerate object detection. 
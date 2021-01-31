# Efficient, high-performance pancreatic segmentation using multi-scale feature extraction

## Abstract
The success of deep learning in recent years has arguably been driven by the availability
of large datasets for training powerful predictive algorithms. In medical applications,
the sensitive nature of the data limits the collection and exchange of large-scale datasets.
Privacy-preserving and collaborative learning systems can enable the successful
application of machine learning in medicine. However, collaborative protocols such as
federated learning require the frequent transfer of parameter updates over a network.
To enable the deployment of such protocols to a wide range of systems with varying
computational performance, efficient deep learning architectures for
resource-constrained environments are required.
Here we present MoNet, a small, highly optimized neural-network-based
segmentation algorithm leveraging efficient multi-scale image features. MoNet is a
shallow, U-Net-like architecture based on repeated, dilated convolutions with decreasing
dilation rates. We apply and test our architecture on the challenging clinical task of
pancreatic segmentation in computed tomography images. We assess our model's
segmentation performance and demonstrate that it provides superior out-of-sample
generalization performance, outperforming larger architectures, while utilizing
significantly fewer parameters. We furthermore confirm the suitability of our
architecture for federated learning applications by demonstrating a substantial reduction
in serialized model storage requirement as a surrogate for network data transfer. Finally,
we evaluate MoNet's inference latency on the central processing unit (CPU) to
determine its utility in environments without access to graphics processing units.
Our implementation is publicly available as free and open-source software.

## Preprint
https://arxiv.org/abs/2009.00872

## MoNet Architecture
![monet_architecture](/images/monet_architecture.png)

![rddc_block](/images/rddc_block.png)


## Results
![inf_performance](/images/inference_time.png)

![dice_performance](/images/dice_performance.png)

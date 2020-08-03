# Efficient, high-performance pancreatic segmentation using multi-scale feature extraction

## Abstract
For artificial intelligence-based image analysis methods to reach clinical applicability, the development of high-performance segmentation algorithms is crucial. However, existent models based on natural images are neither efficient in their parameter use nor optimized for medical imaging. Here we present MoNet, a highly optimized neural-network-based pancreatic segmentation algorithm focused on achieving high performance by efficient multi-scale image feature utilization.

## Methods
We developed MoNet a shallow,U-Net-like architecture based on repeated, dilated convolutions with decreasing dilation rates. The model was trained on publicly available pancreatic computed tomography (CT) scans in the portal-venous phase from the Medical Segmentation Decathlon (196 training and 85 validation scans) and we tested its performance by evaluating the Dice coefficient on 85 manually segmented scans sourced from our institution’s clinical database. We compared the model’s Dice coefficient and inference time against standard architectures (U-Net and Attention U-Net)

## MoNet Architecture
![monet_architecture](/images/monet_architecture.png)

![rddc_block](/images/rddc_block.png)


## Results
![inf_performance](/images/inference_time.png)

![dice_performance](/images/dice_performance.png)

# PET-CT-segmentation
explore segmentation method for PET/CT images.

In the start of a new project i will implment new PET/CT segmentation from different papers.

this is work in progresss 

DATA :  the data i using is dicom converted to nrrd files, so my analyzing on nrrd file. Unfortunately the is private so i cant share it.

first paper : MSAM - "Multimodal Spatial Attention Module for Targeting Multimodal PET-CT Lung Tumor Segmentation" 

arXiv:2007.14728

in short ; use PET images as attention map (train by UNET) on the CT image for emphasize wanted area. 

how; it take the traind PET (= attention map ) and multiplier it with the add values in the upsampling phase in the CT UNET training.

![image](https://user-images.githubusercontent.com/61969606/136341751-aba8e9b8-9c9f-4fb8-ada0-1de21bb526d8.png)






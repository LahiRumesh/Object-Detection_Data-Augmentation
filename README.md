# Object Detection - Image Augmentation

Images augmentation for Object detection tasks

## Getting Started

This pipeline can use to Augment data in CSV Annotation Format

**(CSV Annotation)** Format:

![CSV_format](/utils_data/SS/csv.png)

Make sure that, your csv file in this order.

#### Installation
Clone this repo with:
```bash
git clone git@github.com:LahiRumesh/Object-Detection_Data-Augmentation.git
cd Object-Detection_Data-Augmentation/
```

Install required packages :

```bash
pip install -r requirements.txt
```

## Image Augmentation

Make changes to **user_inputs.ini** , If you want to Rotate Image in 45 degrees set ROTATE_IMAGES=True 

```bash
[Random_Rotate]
ROTATE_IMAGES=True 
ROTATE_VALUE=45
```
Further more If you want to Rotate and Resize the images, set

```bash
[Random_Rotate]
ROTATE_IMAGES=True 
ROTATE_VALUE=45

[Resize]
RESIZE_IMAGES=True 
RESIZE_VALUE=416

```

Make sure that to Select atleast one Augmentation Type

 **Weed Detection and Classification using Deep Learning**
 **Overview**

This project implements a two-stage deep learning pipeline to detect and classify weeds in crop images:
    **Weed Detection** – Determine whether an image contains a crop or weed.
   **Weed Classification** – If the image contains a weed, classify it into one of the following categories:
        Sedge
        Circium
        Chenopodium
        Bluegrass

This system is intended to support precision agriculture and automated weed control applications.

**What We Did**
1. Weed Detection
    A binary classifier model (weed_detector) was trained to identify whether an image shows a weed or a crop.
    Input: Image from test dataset
    Output: Label – Weed or Crop

2. Weed Classification
    If the detection model predicts Weed, the image is passed to a second model (weed_classifier) that classifies the weed into one of four types:
        Sedge
        Circium
        Chenopodium
        Bluegrass

During testing, most weed predictions were classified as Circium and Sedge, which appeared to be the dominant types in the dataset.


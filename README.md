This is my submission for the object detection in an urban environment project.

the movie from the deployed mode is shown in `output_movie.avi`.

I deployed the EfficientDet D1 model. I found it to be the best architecture I examined. 


The remaining of the repo is the following:

- `1_train_model.ipynb`: trained deployed EfficientDet D1 model
- `pipeline.config`: config of the deployed EfficientDet D1 model
- `mobilenet_training/`: director containing training script and config file for SSD MobileNet V2 model
- `rcnn_training/`: director containing training script and config file for Faster R-CNN ResNet50 model
- `2_deploy_model.ipynb`: record of deploying the EfficientDet D1 model
- `output_movie.avi`: movie recor of EfficientDet D1 model deployment

Here is my summary of the results, which is also in `1_train_model.ipynb`:

# results summary
I ran three models: this baseline EfficientDet D1 model, SSD MobileNet V2, and Faster R-CNN ResNet50. 

SSD MobileNet V2 was based on `https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config`

Faster R-CNN ResNet50 was based on `https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config`

In both cases, I altered the configuration file to run in our file structure and project setting given our computational constraints

Each model is compared in terms of accuracy and validation loss. These are based on the average perfomance when evaluating the model with validation data. The precision is reported as the mean average precision over all cases (e.g., the first AP reported after "Accumulating evaluation results..."). The validation loss is reported directly above that chunk of output (in this notebook, in the line `I1006 05:10:07.066834 140026397906752 model_lib_v2.py:1018] #011+ Loss/total_loss: 0.38`). This loss metric is a combination of localization loss, ensuring bounding boxes are in the right locations, and classification loss, ensuring the detected object is correctly classified. 

| Model Architecture | mAP (Accuracy) | Validation Loss |
|-------------------|----------------|-----------------|
| EfficientDet D1 | 0.095 | 0.386 |
| SSD MobileNet V2 | 0.087 | 0.995 |
| Faster R-CNN ResNet50 | 0.060 | 1.319 |

In all cases, the accuracy is too low and the validation loss is too high. This could be improved by training the models for more epochs and increasing the batch size, both of which are out of scope for this problem. Particularly because of the memory requirements of increasing the batch size, which would require a completely different computational setup.

Interestingly, EfficientDet (this baseline model notebook) is the most accurate of the three models (it has the highest accuracy and the lowest validation loss). It has the highest accuracy and the lowest validation loss. This is likely because EfficientDet D1 is a favorable architecture for this system. The SSD MobileNet V2 structure is designed to run with lower latency, which may have speed advantages not reflected in this analysis. The Faster R-CNN ResNet50 is a more advanced architecture, which I speculate may plateau at a better performance given extensive training compute. 

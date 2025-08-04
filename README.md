## MVRNet

### Introduction
We introduce MVRNet, a deep learning framework for detecting and segmenting intracranial aneurysms (IAs) in 3D CTA images, especially small or obscured lesions missed by conventional methods. Key innovations include:

* A feature enhancement technique to improve boundary clarity and reduce noise.

* A multi-view encoder to address occlusion, paired with a refinement decoder for precise predictions.

MVRNet outperforms state-of-the-art methods, boosting the F1-score by 24% and IoU by 16%, with >2× improvement on challenging cases.

<img src="./graphical.png" alt="图片alt" title="MVRNet" style="width:85%" />

### Checkpoints
We provide the pre-trained model of MVRNet in the `ckpt` directory.

### Dataset


## Data preprocessing

### Train
Run command as below.
```shell script
bash run.sh
```

### Inference

Run command as below.
```shell script
bash evaluation.sh
```

we supply an example dataset
```shell script
raws
├── image  # directory of image files
├── mask   # directory of ground truth
├── lesion_bbox.txt  # lesion bbox info 
├── part_train.txt  # train list
├── part_val.txt    # validation list
├── part_test.txt   # my_test list 
```
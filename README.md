# Unsupervised Face Recognition using Tensorflow #

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper 
["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://www.dropbox.com/s/liw9f1m8w4ups5b/20180408-102900.zip?dl=0) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://www.dropbox.com/s/311vlz79miplvn9/20180402-114759.zip?dl=0) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [NITK_K80_trained](https://www.dropbox.com/s/y1l1sl1fy8shj5h/ravi_chkpt.zip?dl=0) | 0.999 | VGGFace2 |  [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [Dlib Face Predictor](https://www.dropbox.com/s/0hc0pjqh8yg0z7u/shape_predictor_68_face_landmarks.dat?dl=0) | Nil | Nil | Nil | [Dlib](http://dlib.net)

## Workflow of the project
Unsupervised face recognition is can be solved by iterative approach of clustering facial encodings obtained from the pre-trained Neural Network [Inception Resnet v1](https://arxiv.org/pdf/1602.07261.pdf). Workflow of project includes, training 
Inception ResNet v1 model with triplet loss function using labelled dataset which includes [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset, consists of total of 453 453 images over 10 575 identities after face detection. 
Some performance improvement has been seen if the dataset has been filtered before training and the best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes. 

### Semi-supervised learning and Clustering ###

* Train model with annotated dataset.(Already trained and pretrained weights are linked above)
* Gather feature vectors from unlabelled face images which are needed to be clustered by passing them into pretrained nerural network. (final_features ==> embeddings). In general, if we pass any image into neural network for prediction final outcome will be class label, but rather do passing image till last layer of neural network, collect features from last before layer, these are called as EMBEDDINGS.
* Pass embeddgings (each embeddings is of 128 dimensional vector) to clustering algorithm which clusters similar faces. "How embeddings of unlabeled face images are clustering together?", Assumption that if Neural Network is trained with millions of images, it can able to differentiate the similar faces. (Core idealogy of the project)
* Every cluster has to be represented by a vector is size 128 dimensions, which is nothing but a centroid of the cluster. Centroid of a cluster is calculated as mean of all points in cluster. Drawback is that if a cluster containing a point (outliers) which is misclassified then, it affects the centroid of cluster.
* Solution is Medoids, Compute medoids of every cluster so that each medoid represents center of a particular cluster. Clustering has to be repeated if new data points are added (if user adds new images to cloud)
* Clustering will give best results after uploading images by user but clustering is computationally costly. 
* Solution : Perform clustering after every T days, in mean time (between 1 to T days) if user uploads any image, whose embeddings are exported from the pretrained neural network, correlated with each and every medoids previosly computed. if result of correlation is more than threshold 't' then, add new face image into that cluster else ignore, which will be sorted out in next 'T' where clustering for whole dataset is performed again.

## Installing required packages

Anaconda native build has to be installed to satisify basic packages needed, same packages which is used in this project can be cloned after installing anaconda basic version 
`Python 3.5.4 |Anaconda custom (64-bit)| | [GCC 7.2.0] on linux` 

### To use the spec file to create an identical environment on the same machine or another machine:

`conda create --name myenv --file spec-file.txt`  - Here myenv is environment name which can be changed into any custom name. This create new enviroment silimar to used at deploying this model. 

### To use the spec file to install its listed packages into an existing environment:

`conda install --name myenv --file spec-file.txt` - To install packages to already existing env run this line.

Other Requirements are listed in `requirements.txt` file.

## Step_1 : Face detection
Faces has to be detected from given whole images. Download Dlib face detector from [link](https://www.dropbox.com/s/0hc0pjqh8yg0z7u/shape_predictor_68_face_landmarks.dat?dl=0) provided. To get more information to run the file named `face_detection_and_alignment.py` use parameter `-h` as help parameter. Run as `python face_detection_and_alignment.py -p shape_predictor_68_face_landmarks.dat -d input_dir -o output_dir`, output directory results in only cropped and aligned face images. Height and Width of square containing face cropped is 160 x 160 , in case of custom cropping size use -w parameter as `-w 190` .

On executing `python face_detection_and_alignment.py -h` we get below information.,

```
usage: face_detection_and_alignment.py [-h] -p SHAPE_PREDICTOR -d IMAGE_DIR
                                       [-w FACE_WIDTH] -o OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  -p SHAPE_PREDICTOR, --shape-predictor SHAPE_PREDICTOR
                        path to facial landmark predictor
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to input parent directory containing sub_dir of
                        images
  -w FACE_WIDTH, --face-width FACE_WIDTH
                        face cropping size
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output directory of images
```

This is how the input directory containing set of random images are organized:
```
/home/stan/Unsupervised Face Recognition/input_dir
├── folder1
│   ├── _Sharon_0006.JPG
│   ├── Ariel_Sharon_0007.png
│   ├── on_0008.jpg
│   ├── 009.png
│   └── 10.jpg
├── Folder2
│   ├── _Schwarzenegger_0006.png
│   ├── enegger_0007.png
│   ├── 008.PNG
│   ├── 9.png
│   └── _0010_.JPG
├── Colin
│   ├── Colin_Powell_0006.png
│   ├── Colin_0007.jgg
...
...
```
### Helper module cr2 to jpg
Input images can be either of types .JPG (or) .jpg (or) .png (or) .PNG, If images where from canon dslr cameras stored in extensions like image_name.CR2 which are raw images. This project has an added a helper file that converts the CR2 to jpg files. 
To execute  `python cr2_to_jpg.py -s input_dir/ -d results_folder/`, where input_dir/ containing images with extensions .CR2 after excetuting this all CR2 images will be converted into .jpg images. For more information use `-h`
as `python cr2_to_jpg.py -h`
 
## Step_2 : Preprocessing
Face detection from dlib is not cent percentage accurate, 4% of artifacts like images of smilies, trees and clouds which resemble like face and most blurry images are cropped as faces and aligned as output of previous face detection algorithm. Preprocessing is done to remove all these artifacts, mostly removes blurry images. Every image is passed into a test of blurring, by calculating variance of laplacian over the image. If variance is more than threshold 't' then, the image is considered to be blurry. As output only clear and non-blurry images are stored in new folder. Run as `python preprocessing.py -i input_dir -o output_dir` and in case of running `python preprocessing.py -h`  results as below,
```
(tensorflow) stan@stan-ThinkStation-D30 ~/$ python preprocessing.py -h

usage: preprocessing.py [-h] -i IMAGE_DIR [-t THRESHOLD] -o OUTPUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE_DIR, --image_dir IMAGE_DIR
                        path to input directory of images
  -t THRESHOLD, --threshold THRESHOLD
                        focus measures that fall below this value will be
                        considered 'blurry'
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        path to output directory of images

```

This is how the input directory containing set of random images are organized, Make sure arrangements of data in input directory before execution.
```
/home/stan/Unsupervised Face Recognition/input_dir

│_ Schwarzenegger_0006.png
│_ enegger_0007.png
│_ 008.PNG
│_ 9.png
│_ 0010_.JPG
│_ Colin_0007.jgg
...
...
```
## step_3 : Clustering

To export embeddings one of arguments to be passed is model directory containing pretrained model checkpoints. From several experiments on training it is concluded that the model trained at NITK using 8 nodes K80 gpu (using HPC) obtains perfect embeddings for clustering algorithm download links is [NITK_K80_trained](https://www.dropbox.com/s/y1l1sl1fy8shj5h/ravi_chkpt.zip?dl=0). In case of memory exhausted error reduce the `--batch_size` parameter (usually happens during usage GPU version of tensorflow) Execute clustering algorithm as below
`python clustering.py --model_dir new_chkpt --input input_dir/ --output results/ --batch_size 32`
For the parameter of `--model_dir` use the folder extracted from link mentioned above (NITK_K80_trained). This is how the input directory containing set of random images are organized:
```
/home/stan/Unsupervised Face Recognition/input_dir
├── folder1
│   ├── _Sharon_0006.JPG
│   ├── Ariel_Sharon_0007.png
│   ├── on_0008.jpg
│   ├── 009.png
│   └── 10.jpg
├── Folder2
│   ├── _Schwarzenegger_0006.png
│   ├── enegger_0007.png
│   ├── 008.PNG
│   ├── 9.png
│   └── _0010_.JPG
├── Colin
│   ├── Colin_Powell_0006.png
│   ├── Colin_0007.jgg
...
...
```
Results will be in output directory containing subdirectorys named from `1 to n`, resulting each and every directory is a cluster. Every cluster is of minimum size 20 .i.e, containing 
minimum face images as 20. This is how the output directory containing set of clustered images are organized, (every folder is cluster of similar faces):
```
/home/stan/Unsupervised Face Recognition/input_dir
├── 1
│   ├── _Sharon_0006.JPG
│   ├── Ariel_Sharon_0007.png
│   ├── on_0008.jpg
│   ├── 009.png
│   └── 10.jpg ...
├── 2
│   ├── _Schwarzenegger_0006.png
│   ├── enegger_0007.png
│   ├── 008.PNG
│   ├── 9.png
│   └── _0010_.JPG ...
├── 3
│   ├── Colin_Powell_0006.png
│   ├── Colin_0007.jgg ...
...
...
```
An overview of chinese whisper clustering algorithm is described in folder `chinese_whispers_sample_demo` , which contains embeddings obtained by passing face images of 2 different persons faces. Embeddings are in form of test_embeddings.csv
run script as `python chinese_whispers.py` and output is obtained as 
`
[['image_file_path_6', 'image_file_path_7', 'image_file_path_8', 'image_file_path_9', 'image_file_path_10', 'image_file_path_11', 'image_file_path_12', 'image_file_path_13', 'image_file_path_14', 'image_file_path_15'], 
['image_file_path_1', 'image_file_path_2', 'image_file_path_3', 'image_file_path_4', 'image_file_path_5']] 
`
Above resulted in list containing 2 more list, which states that after clustering we obtained 2 clusters and their paths are clubbed together as list of list.

## Step_4 : Medoid Selection and Validation

### Export Embeddings
For selection of medoids or validation first steps is to export embeddings and its corresponding paths. To just export embeddgings from the given parent directory containing sub directory of images run script as  
` python export_embeddings.py new_chkpt/ input_dir/ outputfile.csv` , `new_chkpt/` parameter is path to directory containing checkpoint files of pretrained neural network (preferrably use NITK_K80_Trained model), `outputfile.csv` is file name where the embeddings are saved and `input_dir/` parameter is path to parent directory contains files in subfolders as below,
```
/home/stan/Unsupervised Face Recognition/output_dir
├── 1
│   ├── _Sharon_0006.JPG
│   ├── Ariel_Sharon_0007.png
│   ├── on_0008.jpg
│   ├── 009.png
│   └── 10.jpg ...
├── 2
│   ├── _Schwarzenegger_0006.png
│   ├── enegger_0007.png
│   ├── 008.PNG
│   ├── 9.png
│   └── _0010_.JPG ...
├── 3
│   ├── Colin_Powell_0006.png
│   ├── Colin_0007.jgg ...
...
...
```
Output will be csv file named `paths_labels_embeddings_.csv`. These embeddings are used for the calculation of centroid for a cluster (Medoid). Once cluster centroids are calculated, for validating an image before next iteration of clustering after time T, it is needed to be compared (correlated) with all the medoid of clusters (each medoid represents one cluster) to predict new face belongs to which cluster. 

### Compute Medoids
Medoids are the points representing each clusters center, so that, on validation if new face embedding is to be validated into a specific cluster, then it has to be compared (correlated) with the all existing medoids (centers of cluster). If the new face embedding belongs to a cluster then its correlation value will be maximum with that cluster else image may fall into new cluster, which will be fixed on next iteration of clustering performed at interval of time 'T'. Run as `python get_medoids.py paths_labels_embeddings_.csv medoids.csv` on running help results as below, 

```
python get_medoids.py -h
usage: get_medoids.py [-h] input_csv_file output_csv_file

Make sure csv is similar to paths_labels_embeddings pattern

positional arguments:
  input_csv_file   Similar to paths_labels_embeddings.csv uploaded in repo
  output_csv_file  Name of output csv file where embeddings of medoids are
                   saved along labels with extension file_name.csv

```

Refer `medoids.csv` file to check selected centroids for above 151 clusters whose results are obtained from clustering and export embeddings steps.

### Validation of New Face embeddings
For Validation of new faces whether it belongs to any of existing cluster, newly detected faces has to be passed into neural network and embeddings are obtained. For validation (prove how algorithm works effectively) purpose , images to be identified are labelled manually, i.e. folder name is considered as labels. Directory structure of validation images whose embeddings has to be obtained are as below,
```
/home/stan/Unsupervised Face Recognition/validation_dir
├── 0
│   ├── Final_preprocessing/output/face_44292.jpg
│   ├── Final_preprocessing/output/face_44292.jpg
│   ├── Final_preprocessing/output/face_44292.jpg
│   ├── Final_preprocessing/output/face_44292.jpg
│   └── Final_preprocessing/output/face_44292.jpg
├── 2
│   ├── _Schwarzenegger_0006.png
│   ├── enegger_0007.png
│   ├── 008.PNG
│   ├── 9.png
│   └── _0010_.JPG ...
├── unknown
│   ├── Colin_Powell_0006.png
│   ├── Colin_0007.jgg ...
...
...
```
export embeddings executed as `python export_embeddings.py new_chkpt/ validation_dir/ validate_embeddings.csv` , this results the embeddings of csv file `validate_embeddings.csv` . 
To validate each embedding exported with centers of clusters which is stored in `medoids.csv` file, execute `python validate.py medoids.csv validate_embeddings.csv results.csv` , during validation or clustering comparison metric used is correlation, clustering is performed for 50 iterations in general but validation is done as one time evaluation, so that threshold for correlation is increased to 0.75, stating that if and only if the images are highly correlated to medoids, it is clustered to medoids else it is marked as `unknown`, refer `results.csv` file for the validation results.

## Note
Above every function is modularized and it can be incorporrated in other projects. In order to deploy as cloud software, change file I/O as database operations. Tsne an dimensionality reduction technique, better than compared to pca is done for the embeddings obtained. execute tsne as `python tsne.py` , as result clusters can be visualized from 128 dims to 2d.



















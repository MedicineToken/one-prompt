# One-Prompt to Segment All Meical Image

One-Prompt to Segment All Medical Images, or say One-Prompt, combines the strengths of one-shot and interactive methods. In the inference stage, with just one prompted sample, it can adeptly handle the unseen task in a single forward pass.

This method is elaborated in the paper [One-Prompt to Segment All Medical Images](https://arxiv.org/abs/2305.10300).


## A Quick Overview 
<img width="800" height="580" src="https://github.com/KidsWithTokens/one-prompt/blob/main/figs/oneprompt.png">


## Requirement

Install the environment:

``conda env create -f environment.yml``

``conda activate oneprompt``

##  Dataset
### Download the open-source datasets 
We collected 78 **open-source** datasets for training and testing the model. The datasets and their download links are in [here](https://drive.google.com/file/d/1iXFm9M1ocrWNkEIthWUWnZYY2-1l-qya/view?usp=share_link).

### Download the prompts
The prompts corresponding to the datasets can be downloaded [here](https://drive.google.com/file/d/1cNv2WW_Cv2NYzpt90vvELaweM5ltIe8n/view?usp=share_link). Each prompt is saved a json message with the format ``{DATASET_NAME, SAMPLE_INDEX, PROMPT_TYPE, PROMPT_CONTENT}``

## Train
run ``python train.py -net oneprompt -mod one_adpt -exp_name basic_exp -b 64 -dataset oneprompt -data_path *../data* -baseline 'unet'``

## Test Examples

### Melanoma Segmentation from Skin Images (2D)

1. Download ISIC dataset part 1 from https://challenge.isic-archive.com/data/. Then put the csv files in "./data/isic" under your data path. Your dataset folder under "your_data_path" should be like:

ISIC/

     ISBI2016_ISIC_Part1_Test_Data/...
     
     ISBI2016_ISIC_Part1_Training_Data/...
     
     ISBI2016_ISIC_Part1_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part1_Training_GroundTruth.csv
    
2. run: ``python val.py -net oneprompt -mod one_adpt -exp_name One-ISIC -weights *weight_path* -b 1 -dataset isic -data_path ../dataset/isic -vis 10 -baseline 'unet'``
change "data_path" and "exp_name" for your own useage. you can change "exp_name" to anything you want.

You can descrease the ``image size`` or batch size ``b`` if out of memory.

3. Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.

4. Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

In default, everything will be saved at `` ./logs/`` 

### REFUGE: Optic-disc Segmentation from Fundus Images (2D) 
[REFUGE](https://refuge.grand-challenge.org/) dataset contains 1200 fundus images with optic disc/cup segmentations and clinical glaucoma labels. 

1. Dowaload the dataset manually from [here](https://huggingface.co/datasets/realslimman/REFUGE-MultiRater/tree/main), or using command lines:

``git lfs install``

``git clone git@hf.co:datasets/realslimman/REFUGE-MultiRater``

unzip and put the dataset to the target folder

``unzip ./REFUGE-MultiRater.zip``

``mv REFUGE-MultiRater ./data``

2. For training the adapter, run: ``python val.py -net oneprompt -mod one_adpt -exp_name One-REFUGE -weights *weight_path* -b 1 -baseline 'unet' -dataset REFUGE -data_path ./data/REFUGE-MultiRater``
you can change "exp_name" to anything you want.

You can descrease the ``image size`` or batch size ``b`` if out of memory.

## Run on  your own dataset
It is simple to run omeprompt on the other datasets. Just write another dataset class following which in `` ./dataset.py``. You only need to make sure you return a dict with 


     {
                 'image': A tensor saving images with size [C,H,W] for 2D image, size [C, H, W, D] for 3D data.
                 D is the depth of 3D volume, C is the channel of a scan/frame, which is commonly 1 for CT, MRI, US data. 
                 If processing, say like a colorful surgical video, D could the number of time frames, and C will be 3 for a RGB frame.

                 'label': The target masks. Same size with the images except the resolutions (H and W).

                 'p_label': The prompt label to decide positive/negative prompt. To simplify, you can always set 1 if don't need the negative prompt function.

                 'pt': The prompt. e.g., a click prompt should be [x of click, y of click], one click for each scan/frame if using 3d data.

                 'image_meta_dict': Optional. if you want save/visulize the result, you should put the name of the image in it with the key ['filename_or_obj'].

                 ...(others as you want)
     }

## Cite






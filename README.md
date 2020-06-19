# ava_asd
Scripts for a pipeline to:
 - Process video data for the active speaker classification problem (via face detections).  Uses the AVA ActiveSpeaker data set.
 - Train a model (not state of the art).
 - Evaluate the model against the public validation set.

## Active speaker detection problem
Tries to classify who the active speakers are in a video.

## AVA Active Speaker Detection dataset
Contains youtube videos with face track annotation containing bounding box information for a head, along with 
information on whethe the person is speaking.

## Installation

### OS pre-requisites
- pip: https://pypi.org/project/pip/
- ffmpeg 4.0 or later. See http://ubuntuhandbook.org/index.php/2018/10/install-ffmpeg-4-0-2-ubuntu-18-0416-04/ 
for installation instructions if you are using an older version of Ubuntu.
- Python 3.7. See https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/ for instructions on how to
install Python 3.7 on Ubuntu 18.04 (or earlier).
- virtualenv 20 or greater
  - To install: `pip install --upgrade virtualenv`. 
  - To check your version run: `virtualenv --version`

### Python pre-requisites
It is recommended to install ava_asd in a Python virtual environment to prevent conflicts with other packages.

Make sure you are in the ava_asd folder.
```
cd ava_asd
```

Create a virtual environment:
```
virtualenv -p python3.7 venv
```

Activate your virtual environment:
```
source venv/bin/activate
```

Install dependencies:
```
pip3 install -r requirements.txt
```

Install the ava_asd package in your virtual environment if you want to use the command line tools:
```
pip3 install -e .
```

## Downloading data
You can use the download script provided.

Make a folder to download the data into:
```
mkdir data
```

### Annotations
To download annotations,
```
ava-download annotations configs/config.yaml data
```

### Videos
To download the videos,
```
ava-download videos configs/config.yaml data
```

### Configuration
If you are not downloading everything from scratch, you can customise some of the paths in the `config.yaml` file.
For example, 

## Preparing the data

In order to use the downloaded data, we need to:
1. Extract jpgs, and mfccs from the youtube videos.
2. Generate metadata to use in keras for training.
3. [Optionally] apply filtering and re-balancing of the dataset.

### Extracting jpgs and audio
```
ava-extract videos configs/config.yaml data
```

### Extract MFCCs
```
ava-extract mfccs configs/config.yaml data
```

### Extract the annotations from the CSV into a python data structure for use.
```
ava-extract annotations configs/config.yaml data
```

## Training
Before you start training, check that the settings in ```config.yaml``` are what you want to use.  Once you are happy 
with the parameters, run:
```
ava-train configs/config.yaml data
```

If you want to monitor the progress of a long training run, the tensorboard output will be in ```/tmp/Graph```

Use
```
tensorboard --logdir /tmp/Graph
```
to monitor it if you have tensorboard installed.

## Evaluation
To evaluate a model, run the following command, making sure to customise the path to the weights file:
```
ava-evaluate configs/config.yaml data --weights-file your/path/to/weights.hdf5
```

To run the previous evaluation code, supply --legacy as an argument:
```
ava-evaluate configs/config.yaml data --weights-file your/path/to/weights.hdf5 --legacy
```

To evaluate a directory of models and save the results in a CSV file, run the following command:
```
ava-evaluate configs/config.yaml data --weights-path your/path/to/many/weights/
```

# Example model
[Download weights file](https://couch.science/models/example.hdf5)

AP against original public validation set: 0.7151

# References
1. [AVA-ActiveSpeaker: An Audio-Visual Dataset for Active Speaker Detection](https://arxiv.org/abs/1901.01342)
2. [Naver at ActivityNet Challenge 2019 -- Task B Active Speaker Detection (AVA)](https://arxiv.org/abs/1906.10555)

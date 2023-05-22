# Readme

This repository contains the necessary steps to perform a recognition task using the provided model and dataset. Follow the steps below to get started.

## Step 1: Download Model and Dataset

- Download the "model" and "dataset" folders from [this Google Drive link](https://drive.google.com/drive/folders/13UeUIDSr7yQ65lF-vhTPtmwmLgU8V0yQ?usp=sharing).
- Place the downloaded folders in your desired directory.

## Step 2: Prepare the Official Dataset

- Move the official dataset files into the following directories within the "dataset" folder:
  - Training data: `Dataset/train`
  - Validation data: `Dataset/val`

## Step 3: Install Dependencies

- Open a terminal or command prompt.
- Execute the following command to install the required dependencies:
```shell
  pip install -r requirements.txt
```

## Step 4: Preprocess the Data

- Run the following command in the terminal:
```shell
  python loodtxt.py
```
This script will preprocess the data to prepare it for recognition.

## Step 5: Run the Recognition Script

- Execute the following command to start the recognition process:
```shell
  python trace.py
```


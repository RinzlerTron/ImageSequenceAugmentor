#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

# Example: 
# python -m image_sequence_augmentor --data_url=<url_with_file> --label_sequence=<Label1Label2Label3>
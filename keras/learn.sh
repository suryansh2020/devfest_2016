#!/usr/bin/env bash

source_folders=/mfs/replicated/datasets/image_spam/spam_archive_fumera/spam/spam_archive_jmlr/,/mfs/replicated/datasets/image_spam/spam_archive_fumera/spam/personal_image_spam/,/mfs/replicated/datasets/image_spam/spam_archive_fumera/ham/personal_image_ham/
model_name=tivvit_spam_ham
model_path=/home/eva/git/image-analyzer-data/models/

IFS=',' read -r -a folders <<< "$source_folders"
caffe_input_jsons=""

for folder in "${folders[@]}"
do
    IFS='/' read -r -a elems <<< "$folder"
    dataset_name="${elems[-1]}"

    #/home/eva/git/image-analyzer/src/analyze.py -f $folder -s > /home/eva/git/image-analyzer-data/$dataset_name'.json'
    #/home/eva/git/image-analyzer/src/appendCaffeFeatures.py -i /home/eva/git/image-analyzer-data/$dataset_name'.json' -o /home/eva/git/image-analyzer-data/$dataset_name'_caffe.json'
    caffe_input_jsons=/home/eva/git/image-analyzer-data/tivvit/$dataset_name'_caffe.json',$caffe_input_jsons
done

caffe_input_jsons="/home/eva/git/image-analyzer-data/tivvit/personal_image_ham_caffe.json,/home/eva/git/image-analyzer-data/tivvit/personal_image_spam_caffe.json"

echo Learning keras model
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,nvcc.flags=-D_FORCE_INLINES KERAS_BACKEND=theano python3 /home/eva/git/image-analyzer/src/neural.py -f $caffe_input_jsons -s $model_name -d $model_path


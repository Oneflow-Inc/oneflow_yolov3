#!/bin/bash
# # Download Images
# wget -c https://pjreddie.com/media/files/train2014.zip
# wget -c https://pjreddie.com/media/files/val2014.zip

# get label file
tar xzf labels.tgz

# set up image list
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

# copy label txt to image dir
find labels/train2014/ -name "*.txt"  | xargs -i cp {} images/train2014/
find labels/val2014/   -name "*.txt"  | xargs -i cp {} images/val2014/
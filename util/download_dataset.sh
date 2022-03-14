#!/bin/bash

# check aws cli installed
if ! [ -x "$(command -v git)" ]; then
  echo 'AWS cli not installed' >&2
  exit 1
fi

# check output dir provided
if [ ! -n "$1" ]
then
    echo "Usage: `basename $0` output_directory"
    exit
fi

aws s3 --no-sign-request sync s3://open-images-dataset/validation $1

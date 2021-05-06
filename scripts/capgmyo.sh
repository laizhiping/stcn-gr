#!/bin/bash

## CapgMyo dba

# LAST_SUBJECT="18"
# SUBJECT="1"
# while [ $SUBJECT -le $LAST_SUBJECT ]
# do
#     python main.py --config_path ../config/capgmyo.json --subjects $SUBJECT
#     SUBJECT=$[$SUBJECT+1]
# done

python main.py --config_path ../config/capgmyo.json


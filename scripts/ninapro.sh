#!/bin/bash

## Ninapro db1

LAST_SUBJECT="27"
SUBJECT="1"
while [ $SUBJECT -le $LAST_SUBJECT ]
do
    python main.py --config_path ../config/ninapro.json --subjects $SUBJECT
    SUBJECT=$[$SUBJECT+1]
done


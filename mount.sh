#!/bin/bash
s3fs cs230-af-data aws_bucket -o passwd_file=${HOME}/.passwd-s3fs -o url=https://s3.us-west-1.amazonaws.com

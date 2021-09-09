#!/usr/bin/env python
# coding: utf-8

#!pip install --upgrade azureml-sdk azureml-dataprep matplotlib

from azureml.core import Workspace, Dataset
import pandas as pd
import argparse
import azureml.core
from azureml.core import Run
import os

#Get the run context of this run and access the dataset using named inputs
run = Run.get_context()
raw_avro_dataset = run.input_datasets['input_dataset']

#Process the arguments/parameters passed to the pipeline
parser = argparse.ArgumentParser()
parser.add_argument('--minute', type=str, dest='minute_filter', help='path to saved model file')
args = parser.parse_args()


# mount dataset to the compute nodes
mount_context=raw_avro_dataset.mount()
mount_context.start()

# See where we mounted the file dataset
print(mount_context.mount_point)
# List files in there
print(os.listdir(mount_context.mount_point))


#Display Recursive - folder and files 
dataset_mount_folder = mount_context.mount_point
#for root,d_names,f_names in os.walk(dataset_mount_folder):
#	print(root, d_names, f_names)

print("Filter for this run: " + args.minute_filter)

filtered_files = []

for dirpath, dirs, files in os.walk(os.path.join(dataset_mount_folder,'eventhub1/0/2021/09/08/'+args.minute_filter+'/')):
    
    for f in files:
        filtered_files.append(os.path.join(dirpath, f))


#using FastAvro to read AVRO files and converting to a Pandas DataFrame
from fastavro import writer, reader, parse_schema

df = pd.DataFrame()

for f_avro in filtered_files:
    with open(f_avro, 'rb') as fo:
        avro_reader = reader(fo)
        records = [r for r in avro_reader]
    print(len(records))    
    temp_df = pd.DataFrame.from_records(records)
    df = df.append(temp_df)


print(df.head(2))
print(df.shape)

#Process the dataframe and register as new dataset. 

# Unmount dataset
mount_context.stop()


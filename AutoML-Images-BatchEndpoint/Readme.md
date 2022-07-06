#### Please check the article below for CLI commands to work with Managed Batch Endpoints
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-endpoint

1. create batch endpoint - 'az ml batch-endpoint create --name $ENDPOINT_NAME'
2. create deployment - 'az ml batch-deployment create --name nonmlflowdp --endpoint-name $ENDPOINT_NAME --file endpoints/batch/nonmlflow-deployment.yml --set-default'
3. Verify deployment - 'az ml batch-deployment show --name nonmlflowdp --endpoint-name $ENDPOINT_NAME'
4. Invoke with registered file dataset (containing image files) - az ml batch-endpoint invoke --name $ENDPOINT_NAME --input-dataset azureml:<dataset-name>:<dataset-version>
5. Output file by default is stored in the workspaceblobstore. Exact location can be found in the score file generated in the run output/logs folder 

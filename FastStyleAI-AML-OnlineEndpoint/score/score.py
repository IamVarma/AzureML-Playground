import functools
import os
import logging
import json
import numpy
from PIL import Image
from io import BytesIO

#from matplotlib import gridspec
#import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

## Ref Code for FastStyle AI - https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_arbitrary_image_stylization.ipynb

def init():


    
    global storage_url
    global storage_sas
    storage_url = 'https://xyz.blob.core.windows.net/'
    storage_sas = 'SAS Token'

    
    blob_service_client = BlobServiceClient(account_url=storage_url, credential=storage_sas) 
    global container_client
    container_client =  blob_service_client.get_container_client('original-styled-images')

    


    #Define util functions
    global crop_center
    def crop_center(image):
        """Returns a cropped square image."""
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, new_shape, new_shape)

        return image

    global load_image
    @functools.lru_cache(maxsize=None)
    def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads and preprocesses images."""
        # Cache image file locally.
        image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
        # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
        img = tf.io.decode_image(
            tf.io.read_file(image_path),
            channels=3, dtype=tf.float32)[tf.newaxis, ...]
        img = crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

        

        return img

    global tensor_to_image_stream
    
    def tensor_to_image_stream(input_tensor):
        tensor = input_tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        imagefile = BytesIO()
        Image.fromarray(tensor).save(imagefile, format='PNG')
        imagefile.seek(0)
        return imagefile.getvalue()
    
    global style_image_urls
    style_image_urls = ['https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Amadeo_de_Souza-Cardoso_1915.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Derkovits_Gyula_Taligas_1920.jpg',
   'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Derkovits_Gyula_Woman_head_1922.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Edvard_Munch_1893_The_Scream.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/JMW_Turner_Nantes_from_the_Ile_Feydeau.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Derkovits_Gyula_Woman_head_1922.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Edvard_Munch_1893_The_Scream.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/JMW_Turner_Nantes_from_the_Ile_Feydeau.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Large_bonfire.jpg',
      'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Les_Demoiselles_dAvignon.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Pablo_Picasso_1911_Still_Life_with_a_Bottle_of_Rum.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Pablo_Picasso.jpg'
     ]


    ''' 'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Derkovits_Gyula_Woman_head_1922.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Edvard_Munch_1893_The_Scream.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/JMW_Turner_Nantes_from_the_Ile_Feydeau.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Large_bonfire.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Derkovits_Gyula_Woman_head_1922.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Edvard_Munch_1893_The_Scream.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/JMW_Turner_Nantes_from_the_Ile_Feydeau.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Large_bonfire.jpg',
      'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Les_Demoiselles_dAvignon.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Pablo_Picasso_1911_Still_Life_with_a_Bottle_of_Rum.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Pablo_Picasso.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Still_life_1913_Amadeo_Souza-Cardoso.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/The_Great_Wave_off_Kanagawa.jpg',
    'https://mlopsvarmaamlsa.blob.core.windows.net/style-images/Van_Gogh_Starry_Night.jpg'''


    global hub_module
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    logging.info("Init complete")

@rawhttp
def run(request):

    logging.info("Request received")
    print("Request: [{0}]".format(request))
    #data = json.loads(request)["image_urls"]'''
    
    data = json.loads(request.get_data())["image_urls"]) 
    print(data)


    content_image_url = data[0][0]
    print(content_image_url)
    
    
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the 
    # recommended image size for the style image (though, other sizes work as 
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)

    styled_image_tensors = []

    for style_image_url in style_image_urls:

        style_image = load_image(style_image_url, style_img_size)
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        
        styled_image_tensors.append(stylized_image)

    #Upload the images to blob storage containers and the URLs for the image
    i=1
    response_urls = []

    print("upload final styled image")
    for image_tensor in styled_image_tensors:
        
        file_name = content_image_url.split('/')[-1].split('.')[0]+'_styled_'+ str(i)+'.png'
        style_image_blob_client = container_client.get_blob_client(file_name)
        style_image_blob_client.upload_blob(tensor_to_image_stream(image_tensor), blob_type="BlockBlob", overwrite= True)
        i=i+1

        response_urls.append('https://mlopsvarmaamlsa.blob.core.windows.net/original-styled-images/'+file_name)

    logging.info("Request processed")

    if request.method == 'POST':

        print("POST")
        resp = AMLResponse(response_urls,200,json_str=True)
        resp.headers['Access-Control-Allow-Origin'] = "*"

    print(resp.headers)

    return resp
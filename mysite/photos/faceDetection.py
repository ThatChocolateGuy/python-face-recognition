import Algorithmia
import os

import tinify

from mysite.settings import BASE_DIR, ALGORITHMIA_KEY, TINYFY_KEY

# Get directory for file manipulation
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Authenticate Algorithmia with API key
apiKey = ALGORITHMIA_KEY
# Create the Algorithmia client object
client = Algorithmia.client(apiKey)
# Path for client directory
client_dir = "data://ThatChocolateGuy/"

# Authenticate Tinify with API key
tinify.key = TINYFY_KEY


def get_images(image):
    """Create labeled dataset from image collection."""
    local_image_path = DIR_PATH + "/../.." + image.file.url
    image_file_name = image.file.name.split('/')[1]
    client_image_dir = client.dir(client_dir + "PFR/")
    client_image_path = client_dir + "PFR/" + image_file_name
    images = []
    
    # Create remote directory (if it doesn't exist)
    if client_image_dir.exists() is False:
        client_image_dir.create()
    # Upload image to client directory
    client.file(client_image_path).putFile(local_image_path)

    # Retrieve images from data collection
    for file in client_image_dir.list():
        path = file.path.split('_')
        first_name = file.path.split('/')[2].split('_')[0]
        last_name = path[1]
        # Create label from image name
        label = first_name + " " + last_name
        # Label image based on person
        images.append(
            {"url": "data://{0}".format(file.path), "person": label}
        )

    return images


def facial_recgnition_algo(input):
    """Call face recognition algo and pipe input in"""
    algo = client.algo('cv/FaceRecognition/0.2.2')

    print("\nPIPING INPUT:")
    print("\n" + input.__str__() + "\n")
    return algo.pipe(input).result


def train_images(images):
    """Train model from images"""
    input = {
        "action": "add_images",
        "data_collection": "FaceRecognitionStates",
        "name_space": "pfr",
        "images": images
    }
    return facial_recgnition_algo(input)


def model_predictions(image_src):
    """Predict a person based on unseen images using trained model"""
    client_image_dir = client.dir(client_dir + "FaceRecognitionOutput/")
    client_image_path = client_dir + "FaceRecognitionOutput/pfr_out_input.jpg"
    client.file(client_image_path).putFile("." + image_src)

    input = {
        "name_space": "pfr",
        "data_collection": "FaceRecognitionStates",
        "action": "predict",
        "images": [
            {
                "url": "data://ThatChocolateGuy/FaceRecognitionOutput/" +
                "pfr_out_input.jpg",
                "output": "data://ThatChocolateGuy/FaceRecognitionOutput/" +
                "pfr_out_processed.jpg"
            }
        ]
    }

    return facial_recgnition_algo(input)

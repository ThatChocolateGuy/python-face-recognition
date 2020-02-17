import Algorithmia
import os

# Authenticate Algorithmia with API key
apiKey = "simsX9jEbWI8fgSqUz1+vQ+IvGr1"
# Create the Algorithmia client object
client = Algorithmia.client(apiKey)
# Path for client directory
client_dir = "data://ThatChocolateGuy/"
# Get directory for file manipulation
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_images(image):
    """Create labeled dataset from data collection."""
    image_dir = client.dir(client_dir + "PFR/")
    image_file_name = image.file.name.split('/')[1]
    image_file_path = client_dir + "PFR/" + image_file_name
    images = []
    # Create remote directory (if doesn't exist)
    if image_dir.exists() is False:
        image_dir.create()
    # Upload image to client directory
    client.file(image_file_path).putFile(dir_path + "../.." + image.file.url)
    # Retrieve images from data collection
    for file in image_dir.list():
        path = file.path.split('_')
        first_name = file.path.split('/')[2].split('_')[0]
        last_name = path[1]
        # Create label from image name
        label = first_name + " " + last_name
        # Label image based on person
        images.append(
            {"url": "data://{0}".format(file.path), "person": label}
        )

    print("\n" + images.__str__())
    return images


def facial_recgnition_algo(input):
    """Call face recognition algo and pipe input in"""
    algo = client.algo('cv/FaceRecognition/0.2.2')

    print("\n" + input.__str__() + "\n")
    return algo.pipe(input).result
    # return algo.pipe(input)


def train_images(images):
    """Train images from pictures"""
    input = {
        "action": "add_images",
        "data_collection": "FaceRecognitionStates",
        "name_space": "pfr",
        "images": images
    }
    return facial_recgnition_algo(input)


def model_predictions(image_src):
    """Predict unseen images of a person the model was trained on"""
    image_dir = client.dir(client_dir + "FaceRecognitionOutput/")
    image_file_path = client_dir + "FaceRecognitionOutput/pfr_test.jpg"
    client.file(image_file_path).putFile("." + image_src)

    input = {
        "name_space": "pfr",
        "data_collection": "FaceRecognitionStates",
        "action": "predict",
        "images": [
            {
                "url": "data://ThatChocolateGuy/FaceRecognitionOutput/" +
                "pfr_test.jpg",
                "output": "data://ThatChocolateGuy/FaceRecognitionOutput/" +
                "pfrTest.jpg"
            }
        ]
    }

    return facial_recgnition_algo(input)

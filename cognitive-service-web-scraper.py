# +
import os
import requests
import argparse
import cv2
from imutils import paths
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# -

parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--root_dir', type=str, help='Root directory to store photos')
parser.add_argument('--image_dim', type=str, help='Image dimensions')
args = parser.parse_args()

# root_dir = args.root_dir
# image_dim = int(args.image_dim)
root_dir = 'data'
image_dim = 512
train_dir = os.path.join(root_dir, 'train')
valid_dir = os.path.join(root_dir, 'valid')

# +
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)
# -

headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

subscription_key = '63b11f32293f4f73bb805bd7f88e093e'
search_url = 'https://eastus.api.cognitive.microsoft.com/bing/v7.0/images/search'

num_images = 500
search_terms = ['Lebron James', 'Stephen Curry']

for search in search_terms:
    
    class_name = search.replace(' ','_')
    train_class_dir = os.path.join(train_dir, class_name)
    valid_class_dir = os.path.join(valid_dir, class_name)
    
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
        
    if not os.path.exists(valid_class_dir):
        os.makedirs(valid_class_dir)
        
    counter = 0
    train_split = int(num_images*0.7)
    num_searches = int(num_images/150)+1
    
    for i in range(num_searches):
        
        response = requests.get(
            search_url, 
            headers = {
                'Ocp-Apim-Subscription-Key' : subscription_key
            }, 
            params = {
                'q': search, 
                'imageType': 'photo',
                'imageContent': 'Portrait',
                'count': 150,
                'offset': i*150
            })
        response.raise_for_status()
        results = response.json()["value"]

        for image in results:
            if counter > num_images:
                break
            if image['encodingFormat'] == 'jpeg':
                print('Writing image {} for {}...'.format(counter, search))
                directory = os.path.join(train_dir, class_name) if counter < train_split else os.path.join(valid_dir, class_name)
                filename = '{}/{}.jpg'.format(directory, counter)
                try:
                    with time_limit(5):
                        with open(filename, 'wb') as file:
                            download = requests.get(image['contentUrl'], headers=headers)
                            file.write(download.content)
                        counter += 1
                except:
                    print('Skipping {} due to download error:'.format(filename))

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for file in list(paths.list_images(root_dir)):
    try:
        image = cv2.imread(file)
        print('Cropping image: {}'.format(file))
        detected_face = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
        x = detected_face[0][0]
        y = detected_face[0][1]
        l = detected_face[0][2]
        w = detected_face[0][3]
        image = image[y:y+l, x:x+w]
        image = cv2.resize(image, (image_dim, image_dim))
        cv2.imwrite(file, image)
    except:
        print('Removing corrupted file: {}'.format(file))
        os.remove(file)

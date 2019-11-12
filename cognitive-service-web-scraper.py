import os
import requests
import argparse

parser = argparse.ArgumentParser(description='Professional web scraping arg parser')
parser.add_argument('--root_dir', type=str, help='Root directory to store photos')
args = parser.parse_args()

root_dir = args.root_dir

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

subscription_key = '63b11f32293f4f73bb805bd7f88e093e'
search_url = 'https://eastus.api.cognitive.microsoft.com/bing/v7.0/images/search'

num_images = 100
search_terms = ['Stephen Curry', 'Lebron James', 'Kawhi Leonard', 'Kevin Durant', 'James Harden']

for search in search_terms:
    
    directory_name = search.replace(' ','_')
    if not os.path.exists(os.path.join(root_dir, directory_name)):
        os.makedirs(os.path.join(root_dir, directory_name))
        
    response = requests.get(
        search_url, 
        headers = {
            'Ocp-Apim-Subscription-Key' : subscription_key
        }, 
        params = {
            'q': search, 
            'imageType': 'photo',
            'imageContent': 'Face',
            'count': num_images*2
        })
    response.raise_for_status()
    results = response.json()["value"]
    
    counter = 0
    for image in results:
        if counter > num_images:
            break
        if image['encodingFormat'] == 'jpeg':
            print('Writing image {} for {}...'.format(counter, search))
            filename = '{}/{}/{}.jpg'.format(root_dir, directory_name, counter)
            with open(filename, 'wb') as file:
                download = requests.get(image['contentUrl'], headers=headers)
                file.write(download.content)
            counter += 1

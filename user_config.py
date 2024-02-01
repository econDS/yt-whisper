# load user config
import json

with open('user_config.json', 'r') as f:
    user_data = json.load(f)

user_config = {
    'model': user_data['model'],
    'language': user_data['language']
}
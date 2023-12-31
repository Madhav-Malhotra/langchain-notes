import os
import requests
from dotenv import load_dotenv

# Load your Hugging Face key
load_dotenv(".env")
KEY = os.getenv('HUGGINGFACEHUB_API_KEY')


def generate_pet_name(pet_type : str) -> str:
    ''' Generates example pet names for your pet type (ex: dog) '''

    # Visit the Hugging Face Model Hub to see other available model URLs
    URL = "https://api-inference.huggingface.co/models/microsoft/deberta-base"
    headers = {"Authorization": f"Bearer {KEY}"}

    # Make the network request
    response = requests.post(URL, headers=headers, json={
        "inputs": f"My {pet_type}'s name is [MASK]."
    })

    # Extract relevant completion information
    names = list(map(lambda x: x['token_str'], response.json()))
    return names

# run a completion
pet_type = 'snake'
out = generate_pet_name(pet_type)
print(f"Here are five names for your pet {pet_type}", out)
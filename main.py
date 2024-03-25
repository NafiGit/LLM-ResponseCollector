import json
import os
import sys

# Check if dependencies are installed, if not, install them
try:
    from langchain_community.llms import Ollama
except ImportError:
    print("Dependency 'langchain' not found. Installing...")
    os.system("pip install langchain")
    from langchain_community.llms import Ollama

# Function to load JSON data from file safely
def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{file_path}'.")
        sys.exit(1)

# Function to save JSON data to file safely
def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# File paths
prompts_file_path = 'prompts.json'
models_file_path = 'models.json'
response_file_path = 'response.json'

# Load prompts from file
prompts = load_json(prompts_file_path)

# Load model configurations from file
models = load_json(models_file_path)

# Initialize list to store response data
response_data = []

# Iterate over each prompt
for prompt in prompts:
    prompt_text = prompt['prompt']  # Extract the prompt text
    
    # Iterate over each model and its parameters
    for model_name, model_params in models.items():
        top_ps = model_params['top_p']  # List of top_p values
        top_ks = model_params['top_k']  # List of top_k values
        temps = model_params['temp']     # List of temperature values
        
        # Iterate over each combination of parameters
        for top_p in top_ps:
            for top_k in top_ks:
                for temp in temps:
                    # Initialize the model with current parameter combination
                    model = Ollama(
                        model=model_name,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temp
                    )

                    # Invoke the model to get response for the prompt
                    try:
                        model_response = model.invoke(prompt_text)
                    except Exception as e:
                        print(f"Error invoking model '{model_name}': {e}")
                        continue
                    
                    # Collect necessary details for the response
                    response_details = {
                        "prompt": prompt_text,
                        "model": model_name,
                        "top_p": top_p,
                        "top_k": top_k,
                        "temp": temp,
                        "response": model_response
                    }
                    
                    # Append the response details to the response data list
                    response_data.append(response_details)
                    print(response_details)
                    
                    # Save responses to response.json file progressively
                    save_json(response_file_path, response_data)
                    print("Response saved to response.json.")

# Print a message when all responses are saved
print("All responses saved to response.json.")
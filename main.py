import json
from langchain_community.llms import Ollama

# Load prompts from prompts.json file
with open('prompts.json', 'r') as prompts_file:
    prompts = json.load(prompts_file)

# Load model configurations from models.json file
with open('models.json', 'r') as models_file:
    models = json.load(models_file)

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
                    model_response = model.invoke(prompt_text)
                    
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
                    with open('response.json', 'w') as response_file:
                        json.dump(response_data, response_file, indent=4)
                        print("Response saved to response.json.")

# Print a message when all responses are saved
print("All responses saved to response.json.")
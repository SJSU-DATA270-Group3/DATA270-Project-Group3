
import os

# Set the API token directly in the script for testing (not recommended for production)
os.environ['REPLICATE_API_TOKEN'] = ''
# Print the API token to verify it's being read correctly
api_token = os.getenv('REPLICATE_API_TOKEN')
print(api_token)

import random
import csv
import replicate
import json
import datetime

# Load JSON data
def load_data(path,iteration):
    with open(path, 'r') as file:
        data = json.load(file)
    return data[iteration]  # Assuming we're using the first iteration for classification

# Function to construct the classification prompt
def get_classification_prompt(data, classes_to_use):
    prompt = "Please classify the following image. Here are the descriptions of each defect class for reference:\n\n"
    for defect_class in classes_to_use:
        class_description = data[defect_class]["Class Description"]
        prompt += f"{data[defect_class]['Group']}: {class_description}\n\n"
    prompt += "\nComplete the following sentence: the following image belongs to Group {}"
    return prompt

# Perform classification
def classify_images(data, test_images, classes_to_use):
    images_to_exclude = []
    for defect_class in classes_to_use:
        images_to_exclude.extend(data[defect_class]['Image Files'])

    results = []

    for image_path in test_images:
        prompt = get_classification_prompt(data, classes_to_use)
        image = open(image_path, "rb")

        start_time = datetime.datetime.now()
        output = replicate.run(
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            input={
                "image": image,
                "prompt": prompt
            }
        )
        output = "".join(output)
        end_time = datetime.datetime.now()

        print(prompt)
        print(output)

        inference_time = (end_time - start_time).total_seconds()
        input_tokens = len(prompt) / 4 + 255  # 255 tokens for the image
        output_tokens = len(output) / 4
        tokens_per_second = (input_tokens + output_tokens) / inference_time

        defect_class = os.path.dirname(image_path)
        defect_class = os.path.basename(defect_class)

        results.append({
            'Actual Class': defect_class,
            'LLaVA Output': output,
            'Inference Time': inference_time,
            'Input Tokens': input_tokens,
            'Output Tokens': output_tokens,
            'Tokens Per Second': tokens_per_second,
            'Image': image_path,
            'Note': user_note
        })
        image.close()

    return results

# Save results to CSV
def save_results_to_csv(results, filename='classification_results.csv'):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

# Define paths and parameters
path = r'images'
data_path = rf'{path}\training_results_multiclass.json'
test_images = json.load(open(rf'{path}\test_images_multiclass.json', 'r'))
classes_to_use = ['scratches','patches', 'rolled-in_scale','crazing','inclusion','pitted_surface']  # List of classes for testing
user_note = 'multiclass fill-in-blank'  # User-defined note for the CSV

# Main execution
iteration = 'iteration-8'
data = load_data(data_path,iteration)
results = classify_images(data, test_images, classes_to_use)
save_results_to_csv(results, filename = f"{path}\classification_results_{user_note}.csv")

print("Classification completed and results saved.")
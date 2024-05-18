
import os

os.environ['REPLICATE_API_TOKEN'] = ''
api_token = os.getenv('REPLICATE_API_TOKEN')
print(api_token)

# Now run your Replicate code
import replicate
import base64
import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_training_images(dir_path, save_path, defect_class_name):
    # Get a list of all files in the directory
    all_files = os.listdir(dir_path)
    # Randomly select 8 images
    selected_files = random.sample(all_files, 1)
    # Initialize a list to store the images
    images = []
    # Read each image and append to the list
    for file in selected_files:
        img_path = os.path.join(dir_path, file)
        img = cv2.imread(img_path)
        images.append(img)
    # Create a white image of size 10xheight for the gap
    gap = np.ones((images[0].shape[0], 10, 3), np.uint8) * 255
    # Add the gap image between each image
    images_with_gaps = []
    for img in images:
        images_with_gaps.append(img)
        images_with_gaps.append(gap)
    # Remove the last gap
    images_with_gaps = images_with_gaps[:-1]
    # Concatenate all images horizontally
    training_img = cv2.hconcat(images_with_gaps)

    # Save combined training image
    combined_img_filename = f"{defect_class_name}_combined_training_image.jpg"
    combined_img_path = os.path.join(save_path, combined_img_filename)
    cv2.imwrite(combined_img_path, training_img)

    return combined_img_path, [os.path.join(dir_path, file) for file in selected_files]

def calculate_text_similarity(texts):
    texts = [str(text) for text in texts]  # Convert all items to strings if they aren't already
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)

# Load or initialize iteration data
def load_or_initialize(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)  # Load the data and store it in a variable
            # Calculate the maximum iteration number from the keys in the loaded data
            max_iteration = max(int(key.split('-')[1]) for key in data.keys()) if data else 0
            return data, max_iteration
    else:
        return {}, 0
    
def create_llava_input_prompt(group_name, current_description, all_descriptions = None, is_first_iteration=True):
    if is_first_iteration == True:
        prompt = (
                    "I am providing several example images that represent a defect class "
                    "labeled as '" + group_name + "' in our study of hot-rolled steel sheet defects. "
                    "These images are specifically selected to highlight the unique features and "
                    "patterns that define this group.\n\n"
                    "Please review the images closely and provide a detailed description of what "
                    "you observe that characterizes '" + group_name + "'. This description should "
                    "include any common patterns, textures, or characteristics you observe that can "
                    "distinguish '" + group_name + "' from other groups. Your description should be "
                    "clear and detailed, enabling it to be used alongside other group descriptions to "
                    "classify an unlabeled image in a future chat. \n\n"
                    "Ensure that your output clearly references '" + group_name + "' and captures "
                    "all pertinent details that make this group unique."
                )
    elif is_first_iteration == False:
        other_descriptions = [f"{key}: {value}" for key, value in all_descriptions.items() if key != group_name]
        prompt = (
            f"Below are the descriptions of defects identified in hot-rolled steel sheet defects study:\n"
            f"{' '.join(other_descriptions)}\n\n"
            f"The current description for {group_name} is: {current_description}\n\n"
            f"Please review the combined image provided and modify the description for {group_name} to better "
            "differentiate it from the other groups based on the features visible in the image."
        )
    return prompt

def get_group_name(index):
    return f"Group {chr(65 + index)}"  # ASCII 'A' is 65

path = r"images"
training_images_path = os.path.join(path, "training_images")
os.makedirs(training_images_path, exist_ok=True)

defect_paths = {
    "scratches": rf"{path}\scratches",
    "rolled-in_scale": rf"{path}\rolled-in_scale",
    "pitted_surface": rf"{path}\pitted_surface",
    "patches": rf"{path}\patches",
    "inclusion": rf"{path}\inclusion",
    "crazing": rf"{path}\crazing"
}

data, current_iteration = load_or_initialize(rf"{path}\training_results_multiclass.json")
total_iterations = 20

if current_iteration >= total_iterations:
    print("Iterations completed")
else:
    for iteration in range(current_iteration + 1, total_iterations + 1):
        print(f"iteration: {iteration}")
        iteration_key = f"iteration-{iteration}"
        data[iteration_key] = {}

        all_descriptions = {f"Group {chr(65 + index)}": data[f"iteration-{iteration - 1}"][dc]["Class Description"] if iteration > 1 else ""
                            for index, dc in enumerate(defect_paths)}

        for index, (defect_class, defect_path) in enumerate(defect_paths.items()):
            print(f"iteration: {iteration}, class: {defect_class}")
            group_name = f"Group {chr(65 + index)}"
            current_description = data[f"iteration-{iteration - 1}"][defect_class]["Class Description"] if iteration > 1 else ""

            if iteration == 0: 
                prompt = create_llava_input_prompt(group_name, current_description, all_descriptions=None, is_first_iteration=True)
                combined_img_path, image_files = get_training_images(defect_path,training_images_path,defect_class)
            else: 
                prompt = create_llava_input_prompt(group_name, current_description, all_descriptions=all_descriptions, is_first_iteration=False)
                combined_img_path = data[f"iteration-1"][defect_class]['Combined Image Path']  # use existing image
            
            training_image = open(combined_img_path, "rb")

            start_time = time.time()
            # Call replicate API with the image and prompt
            output = replicate.run(
                "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
                input={
                    "image": training_image,
                    "prompt": prompt
                }
            )

            output = "".join(output)
            end_time = time.time()

            print(group_name)
            print(defect_class)
            print(combined_img_path)
            print(output)
            print("")
            

            inference_time = end_time - start_time
            input_tokens = len(prompt) / 4 + 765  # Assuming 765 tokens for the image
            output_tokens = len(output) / 4
            tokens_per_second = (input_tokens + output_tokens) / inference_time

            if iteration == 1:
                data[iteration_key][defect_class] = {
                    "Group": group_name,
                    "Directory": defect_paths[defect_class],
                    "Combined Image Path": combined_img_path,
                    "Class Description": output,
                    "Image Files": image_files,
                    "Inference Time": inference_time,
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens,
                    "Tokens Per Second": tokens_per_second
                }
            else:
                data[iteration_key][defect_class] = {
                    "Group": group_name,
                    "Directory": defect_paths[defect_class],
                    "Combined Image Path": combined_img_path,
                    "Class Description": output,
                    "Image Files": data["iteration-1"][defect_class]['Image Files'],
                    "Inference Time": inference_time,
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens,
                    "Tokens Per Second": tokens_per_second
                }

            training_image.close()
        
        # Calculate text similarity index after all descriptions are updated
        descriptions = [data[iteration_key][dc]["Class Description"] for dc in defect_paths]
        similarity_matrix = calculate_text_similarity(descriptions)
        for i, dc1 in enumerate(defect_paths):
            for j, dc2 in enumerate(defect_paths):
                if dc1 != dc2:
                    data[iteration_key][dc1][f"{dc1}-{dc2}"] = similarity_matrix[i][j]

        # Save updated data
        with open(rf"{path}\training_results_multiclass.json", 'w') as f:
            json.dump(data, f, indent=4)
            

print("Training data has been processed and saved.")

import matplotlib.ticker as mticker
# Plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
max_similarity_indices = []

for i, defect_class in enumerate(defect_paths):
    ax = axs[i//3, i%3]
    similarities = []
    for iteration in range(1, total_iterations + 1):
        similarities.append([data[f"iteration-{iteration}"][defect_class][f"{defect_class}-{other}"] for other in defect_paths if other != defect_class])
    similarities = np.array(similarities)
    max_similarity_indices.append(np.max(similarities, axis=1))
    
    for j in range(similarities.shape[1]):
        class_list = list(defect_paths.keys())
        class_list.remove(defect_class)
        ax.plot(range(1, total_iterations + 1), similarities[:, j], label=f"{defect_class}-{class_list[j]}")
    
    ax.set_title(f'Similarity Indices for {defect_class}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Text Similarity Index')
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
plt.show()
fig.clf()

for iteration in range(1, total_iterations + 1):
    # Plot max similarity index
    for i, defect_class in enumerate(defect_paths):
        if iteration == 1:
            plt.plot(range(1, total_iterations + 1), max_similarity_indices[i], label=defect_class)
        else:
            plt.plot(range(1, total_iterations + 1), max_similarity_indices[i])

plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Maximum Text Similarity Indices Across Classes')
plt.xlabel('Iteration')
plt.ylabel('Max Text Similarity Index')
plt.legend()
plt.show()
fig.clf()

# Initialize lists to store data for plotting
total_tokens_per_iteration = []
total_inference_time_per_iteration = []
tokens_per_second_per_iteration = []

for iteration in range(1, total_iterations + 1):
    iteration_key = f"iteration-{iteration}"
    total_tokens = sum(data[iteration_key][dc]["Input Tokens"] + data[iteration_key][dc]["Output Tokens"] for dc in defect_paths)
    total_inference_time = sum(data[iteration_key][dc]["Inference Time"] for dc in defect_paths)
    tokens_per_second = total_tokens / total_inference_time if total_inference_time > 0 else 0

    total_tokens_per_iteration.append(total_tokens)
    total_inference_time_per_iteration.append(total_inference_time)
    tokens_per_second_per_iteration.append(tokens_per_second)

# Plotting the total number of tokens per iteration
plt.figure(figsize=(10, 5))
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.plot(range(1, total_iterations + 1), total_tokens_per_iteration, marker='o', linestyle='-', color='b')
plt.title('Total Number of Tokens per Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Total Tokens')
plt.show()

# Plotting the total inference time and tokens per second per iteration with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Iteration Number')
ax1.set_ylabel('Total Inference Time (s)', color=color)
ax1.plot(range(1, total_iterations + 1), total_inference_time_per_iteration, marker='o', linestyle='-', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Tokens per Second', color=color)
ax2.plot(range(1, total_iterations + 1), tokens_per_second_per_iteration, marker='x', linestyle='--', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


plt.title('Inference Time and Token Rate per Iteration')

plt.show()

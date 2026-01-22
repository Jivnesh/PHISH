import re
import os
import argparse
from helper import *
responses = []
ct = 0
## Commnad to run this file
## python Verify_Extraction.py --run_id 1 --print_errors 1
def get_file_id():
    parser = argparse.ArgumentParser(description="Configurable settings via command line.")
    # Add arguments with choices

    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="File id"
    )
    parser.add_argument(
        "--print_errors",
        type=int,
        required=True,
        help="File id"
    )
    return parser
parser = get_file_id()
args = parser.parse_args()
score_trajectory = []
SCORES = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "UNK": 3}
traits = {"O": [], "C": [], "E": [], "A": [], "N": []}
count = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "UNK": 0}
file_id = args.run_id
print_errors = args.print_errors
folder_path = "../Results/Character-BFI/Twilight Sparkle/With_Context/"
# all_files = os.listdir("../Results/llama-3.1-8b-instant/Without_Context/")
all_files = [f for f in os.listdir(folder_path) if '_radar_' not in f and 'trajectories' not in f ]
print(f"All files Number: {len(all_files)}")
file_path = folder_path + all_files[file_id]
print(all_files[file_id])
old_ocean_scores, old_score_trajectory = extract_ocean_and_trajectory(file_path)
def extract_first_number(text):
    """
    Extracts the first occurrence of a number (1-5) written in words from a given text.
    Returns the corresponding integer or None if no match is found.
    """
    pattern = r"\b(one|two|three|four|five)\b"
    word_to_num = {
        "one": "E",
        "two": "D",
        "three": "C",
        "four": "B",
        "five": "A"
    }

    match = re.search(pattern, text, re.IGNORECASE)
    return word_to_num[match.group(0).lower()] if match else None
def extract_pattern_from_file(file_path):
    """
    Reads a text file line by line and extracts the 'Question', 'Label', 'key', and 'Response' fields.
    The 'Response' field spans multiple lines and ends at '+ Conversation History:'.

    Reports:
    1. Total number of lines matching the pattern.
    2. Lines where keywords are present but the pattern doesn't match.

    :param file_path: Path to the input text file.
    """
    # Keywords to check in the line
    keywords = ["Question:", "Label:", "key:"]
    matched_count = 0  # Counter for lines matching the pattern
    mismatched_lines = []  # List to store lines with keywords but no valid pattern
    tracker = 0
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if all(keyword in line for keyword in keywords):
                try:
                    # Extract the Question, Label, and key
                    question = line.split("Question:")[1].split(",")[0].strip()
                    label = line.split("Label:")[1].split(",")[0].strip()
                    key = int(line.split("key:")[1].split(",")[0].strip())
                    
                    # Initialize Response extraction
                    response = None
                    if i + 1 < len(lines) and "+ Response:" in lines[i + 1]:
                        i += 1  # Move to the line containing '+ Response:'
                        response_lines = []
                        while i < len(lines) and "+ Conversation History:" not in lines[i]:
                            response_lines.append(lines[i].strip())
                            i += 1
                        response = " ".join(response_lines).strip()

                    # Create the dictionary
                    extracted_data = {
                        'Question': question,
                        'Label': label,
                        'key': key,
                        'Response': response.replace("+ Response: ","").strip()
                    }
                    reverse_scoring = key == 1  # Key of -1 indicates reverse scoring
                    response = response.replace("+ Response: ","").strip()
                    ## Last version: r'[A-E]\)'
                    # selected_option = re.search(r'\b[A-E]\)\b|\b[A-E]\b', response)
                    selected_option = extract_first_number(response)
                    print(f"Response: {response}")
                    print(f"Selected Option: {selected_option}")
                    # print(f"Raw extraction: {selected_option.group(0)}")
                    # selected_option = selected_option.group(0).replace(")","") if selected_option else "UNK"
                    # reverse_scoring = key
                    score = SCORES[selected_option]
                    count[selected_option] += 1
                    print(f"Reverse Scoring: {reverse_scoring}")
                    if reverse_scoring:
                        traits[label].append(6 - score)
                        # Append score to trajectory
                        score_trajectory.append(6 - score)
                    else:
                        traits[label].append(score)
                        # Append score to trajectory
                        score_trajectory.append(score)
                    if old_score_trajectory[matched_count] != score_trajectory[matched_count]:
                        tracker =  tracker + 1
                        if print_errors == 1:
                            print(f"Line {i}: {extracted_data}")
                            print(f"Choice: {selected_option}")
                            print(f" old: {old_score_trajectory[matched_count]}, new:{score_trajectory[matched_count]}")
                            print("-----------------------")
                    if print_errors != 1:
                            print(f"Line {i}: {extracted_data}")
                            print(f"Choice: {selected_option}")
                            print(f" old: {old_score_trajectory[matched_count]}, new:{score_trajectory[matched_count]}")
                            print("-----------------------")
                    matched_count += 1
                    
                    
                except (IndexError, ValueError) as e:
                    mismatched_lines.append((i + 1, line.strip()))
            i += 1

        # Report results
        print(f"\nTotal lines matching the pattern: {matched_count}")
        print(f"Total errors: {tracker}")
        if mismatched_lines:
            print("\nLines with keywords but pattern mismatch:")
            for line_number, line in mismatched_lines:
                print(f"Line {line_number}: {line}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage



# Example usage
# Replace 'input.txt' with the path to your file

extract_pattern_from_file(file_path)


# Define scoring and initialize data structures




# # Process and evaluate responses
# for response in responses:
#     choice = response["response"]
#     count[choice] += 1
#     trait = response["trait"]
#     reverse_scoring = response["reverse_scoring"]
#     score = SCORES[choice]

#     if reverse_scoring:
#         traits[trait].append(6 - score)
#         # Append score to trajectory
#         score_trajectory.append(6 - score)
#     else:
#         traits[trait].append(score)
#         # Append score to trajectory
#         score_trajectory.append(score)

# Calculate mean and variance
def calc_mean_and_var(result):
    mean = {key: np.mean(np.array(item)) for key, item in result.items()}
    std = {key: np.std(np.array(item)) for key, item in result.items()}
    return mean, std

mean, std = calc_mean_and_var(traits)
print("-----------------------------")
print(all_files[file_id])
print("-----------------------------")
print("\nOCEAN Scores (Mean and Std Dev):")


print(f"OCEAN mean Scores: {[float(value) for value in mean.values()]}")

print(f"OCEAN std deviation: {[float(value) for value in std.values()]}")
for trait in traits.keys():
    print(f"{trait}: Mean = {mean[trait]:.2f}, Std Dev = {std[trait]:.2f}")
  

print("\nOption Counts:")

print(count)


if old_score_trajectory == score_trajectory:
    print("All matched")
else:
    print("Fixed the issue with the extraction")
    print(f"\nScore Trajectory: {score_trajectory}")
    
    print(f"\n Old Score Trajectory: {old_score_trajectory}")

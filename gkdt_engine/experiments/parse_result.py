import os
import re
import csv

def extract_accuracies_from_file(file_path, dataset_names):
    """
    Extract accuracy information from the output file.
    
    Args:
        file_path: Full path to the output file.
        dataset_names: List of dataset names.
    
    Returns:
        list: A list of accuracy values.
    """
    accuracies = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Use regex to match all qualifying result lines
            pattern = r'episode 1000/1000, Acc \[([0-9.]+)\]'
            matches = re.findall(pattern, content)
            
            # Only take the first 23 matches (corresponding to 23 datasets)
            matches = matches[:23]
            
            if len(matches) != len(dataset_names):
                print(f"Warning: found {len(matches)} results in {file_path}, but expected {len(dataset_names)} datasets")
            
            # Process each matched result
            for i, match in enumerate(matches):
                if i < len(dataset_names):
                    acc_value = float(match)
                    # Convert to percentage with two decimal places, rounded
                    acc_percent = round(acc_value * 100, 2)
                    accuracies.append(acc_percent)
                else:
                    break
                    
    except FileNotFoundError:
        print(f"Error: file {file_path} does not exist")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
    return accuracies

def save_results_to_csv(all_results, output_file_path, dataset_names, filenames):
    """
    Save all results to a CSV file (horizontal layout).
    
    Args:
        all_results: List of results for all files, where each element is an accuracy list.
        output_file_path: Output file path.
        dataset_names: List of dataset names.
        filenames: List of filenames.
    """
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header: filename in the first column, followed by dataset names
            header = ['Filename'] + dataset_names
            writer.writerow(header)
            
            # Write data rows
            for i, (filename, accuracies) in enumerate(all_results):
                if len(accuracies) == len(dataset_names):
                    row = [filename] + accuracies
                    writer.writerow(row)
                else:
                    print(f"Warning: the number of results for file {filename} does not match, skipping")
        
        print(f"CSV results saved to: {output_file_path}")
        
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def process_eval_files(root_dir, filenames, dataset_names):
    """
    Process all evaluation files and generate a horizontally formatted CSV.
    
    Args:
        root_dir: Root directory path.
        filenames: List of filenames.
        dataset_names: List of dataset names.
    """
    all_results = []  # Store all file results
    
    for filename in filenames:
        file_path = os.path.join(root_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue
        
        print(f"Processing file: {filename}")
        
        # Extract accuracy information
        accuracies = extract_accuracies_from_file(file_path, dataset_names)
        
        if accuracies:
            all_results.append((filename, accuracies))
            
            # Display the result for this file in the console
            print(f"Results for {filename}:")
            for j, dataset_name in enumerate(dataset_names):
                if j < len(accuracies):
                    print(f"  {dataset_name}: {accuracies[j]:.2f}%")
            print()
        else:
            print(f"No valid results found in file {filename}\n")
    
    # If there are results, save to a CSV file
    if all_results:
        output_file_path = os.path.join(root_dir, "all_results.csv")
        save_results_to_csv(all_results, output_file_path, dataset_names, filenames)
        
        # Display a preview of the CSV contents
        print("\nCSV content preview:")
        print("Filename", end="")
        for name in dataset_names:
            print(f",{name}", end="")
        print()
        
        for filename, accuracies in all_results:
            print(filename, end="")
            for acc in accuracies:
                print(f",{acc:.2f}", end="")
            print()
    else:
        print("No valid results were found")

# Example usage
if __name__ == "__main__":
    # Configure parameters
    roots = [
        "/your/path/to/General-Keypoint-Detection/experiments/study_archs/eval_GKDT_L",
    ]
    
    filenames = [ "eval_1shot.out", "eval_0shot.out", "eval_1shot+text.out"]
    # filenames = [ "eval_1shot+text.out"]
    
    # Dataset name list passed as a parameter (22 test sets + 1 additional novel set)
    dataset_names = [
        "Animal pose", "AwA", "CUB", "NAB", "ap10k test", 
        "vinegar fly", "locust", "topviewmouse5k", "macaque", 
        "atrw tiger", "ak test", "coco val", "human art", 
        "human face 300w", "animalweb", "onehand10k", "HInt", 
        "keypoint-5", "carfusion", "df2 val", "cephalometric", 
        "hand xray (base kp)", "hand xray (novel kp)"
    ]
    
    # Process files with the dataset name list passed in
    for i, each_root in enumerate(roots):
        print(f"==>root {i}: {each_root}")
        process_eval_files(each_root, filenames, dataset_names)
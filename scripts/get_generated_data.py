import os
import shutil

def clone_filtered_folders(root_folder, destination_folder):
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path) or not subfolder.isdigit():
            continue
        
        for subsubfolder in os.listdir(subfolder_path):
            if not subsubfolder.startswith(f"{subfolder}."):
                continue
            
            try:
                index = float(subsubfolder.split(".")[1])
                if not (1 <= index <= 5):
                    continue
            except (ValueError, IndexError):
                continue
            
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            if not os.path.isdir(subsubfolder_path):
                continue
            
            caption_path = os.path.join(subsubfolder_path, "caption.txt")
            
            if not os.path.isfile(caption_path):
                continue
            
            dest_subfolder_path = os.path.join(destination_folder, subfolder, subsubfolder)
            os.makedirs(dest_subfolder_path, exist_ok=True)
            
            dest_caption_path = os.path.join(dest_subfolder_path, "caption.txt")
            if os.path.isfile(dest_caption_path):
                print(f"Skipped {caption_path}, already exists in {dest_subfolder_path}")
                continue
            
            shutil.copy(caption_path, dest_subfolder_path)
            print(f"Copied {caption_path} to {dest_subfolder_path}")

# Example usage
root_folder = "/home/ltnghia02/dataset"  # Change to your actual root folder
destination_folder = "/home/ltnghia02/generated_data_30_40/" # Change to your desired destination path
clone_filtered_folders(root_folder, destination_folder)


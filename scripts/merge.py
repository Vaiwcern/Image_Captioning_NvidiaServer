import os
import shutil
from datetime import datetime, timezone, timedelta

current_time = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d-%Hh%M")

INDEX_RANGE = (0, 10000)
SUBFOLDER_SUFFIXES = (".1", ".2", ".3", ".4", ".5")

SOURCE_DIR = "4238-10000-res-step1"
DESTINATION_DIR = f"dataset"

# os.makedirs(DESTINATION_DIR, exist_ok=True)

for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    # Check article index in range
    if os.path.isdir(folder_path) and folder.isdigit() and INDEX_RANGE[0] <= int(folder) <= INDEX_RANGE[1]:
        for subfolder in os.listdir(folder_path):
            # Check image index
            if subfolder.endswith(SUBFOLDER_SUFFIXES):
                subfolder_path = os.path.join(folder_path, subfolder)

                # Skip if DONE.txt exists
                # done_file_path = os.path.join(subfolder_path, "DONE.txt")
                # if os.path.isfile(done_file_path):
                #     continue

                # Skip subfolders with only 2 files = NOT RUN YET, caption.txt exist
                # if os.path.isdir(subfolder_path) and len(os.listdir(subfolder_path)) > 2 and "caption.txt" in os.listdir(subfolder_path):

                if os.path.isdir(subfolder_path) and "molmo_dense_cap.txt" in os.listdir(subfolder_path):
                    dest_path = os.path.join(DESTINATION_DIR, folder, subfolder)
                    # os.makedirs(dest_path, exist_ok=True)

                    # with open(done_file_path, "w") as done_file:
                    #     done_file.write("This folder has been processed.\n")  

                    for file in os.listdir(subfolder_path):
                        if file.endswith("molmo_dense_cap.txt"):
                            file_path = os.path.join(subfolder_path, file)
                            shutil.move(file_path, dest_path)

                    

        print(f"DONE {folder}")
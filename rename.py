import os

def rename_files(folder_path, prefix="file", start_index=1):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    files = sorted(os.listdir(folder_path))  # Sort files to maintain order
    index = start_index

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):  # Ensure it's a file, not a folder
            file_ext = os.path.splitext(file_name)[1]  # Get file extension
            new_name = f"{prefix}_{index}{file_ext}"  # New name format
            new_path = os.path.join(folder_path, new_name)

            # Ensure no overwriting
            while os.path.exists(new_path):
                index += 1
                new_name = f"{prefix}_{index}{file_ext}"
                new_path = os.path.join(folder_path, new_name)

            os.rename(file_path, new_path)
            print(f"Renamed: {file_name} → {new_name}")
            index += 1

if __name__ == "__main__":
    folder = input("Enter folder path: ")
    prefix = input("Enter filename prefix (default: 'file'): ") or "file"
    start_index = int(input("Enter starting index (default: 1): ") or 1)

    rename_files(folder, prefix, start_index)

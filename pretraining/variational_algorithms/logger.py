import json
import os


class Logger:
    def __init__(self, folder_path, file_name=None) -> None:
        if file_name is None:
            
            os.makedirs(folder_path, exist_ok=True)

            file_path = os.path.join(folder_path, "metadata.json")
            
            try:
                with open(file_path, "r+", encoding="utf-8") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                
                metadata = {"name_id": 0}

            
            if "name_id" in metadata:
                file_name = f"log_{metadata['name_id']}.json"
                metadata["name_id"] += 1
            else:
                metadata["name_id"] = 0
                file_name = f"log_{metadata['name_id']}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f)

        self.path = f"{folder_path}/{file_name}"
        self.log = open(self.path, "a+", encoding="utf-8")
        self.log.write("[")

    def write(self, data: dict):
        
        self.log.write(json.dumps(data))
        self.log.write(",")

    def close(self):
        
        self.log.seek(self.log.tell() - 1)  # Move the cursor to the last character
        last_char = self.log.read(1)
        if last_char == ",":
            self.log.seek(self.log.tell() - 1)  # Move the cursor back
            self.log.truncate()  # Remove the trailing comma
        self.log.write("]")
        self.log.close()

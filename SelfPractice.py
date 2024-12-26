import json

def update_image_file_name(source_file, target_file, output_file):
    # Load the source and target JSON files
    with open(source_file, 'r') as src:
        source_data = json.load(src)
    with open(target_file, 'r') as tgt:
        target_data = json.load(tgt)

    # Flatten target data into a dictionary for fast lookup
    target_lookup = {item["ModelName"]: item for item in target_data.get("Monitors", [])}

    # Iterate over the source data and update ImageFileName where it is LINEART
    for monitor_type, monitor_list in source_data.items():
        for monitor in monitor_list:
            model_name = monitor.get("ModelName")
            if model_name and monitor.get("ImageFileName") == "LINEART":
                target_monitor = target_lookup.get(model_name)
                if target_monitor:
                    monitor["ImageFileName"] = target_monitor.get("ImageFileName", "LINEART")

    # Save the updated source data to the output file
    with open(output_file, 'w') as out:
        json.dump(source_data, out, indent=4)
    print(f"Updated source data saved to {output_file}")

source_json_file = "LSTDDPM.json"
target_json_file = "SustainMonitors.json"
output_json_file = "updated_source.json"

update_image_file_name(source_json_file, target_json_file, output_json_file)
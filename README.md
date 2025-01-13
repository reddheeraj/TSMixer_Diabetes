# TSMixer based Model for Hyper and Hypo Glucose Events

## Preprocessing
- Load the Ohio_Data in the following structure:

```
Ohio_Data
|-- OhioT1DM-training
|-- OhioT1DM-testing
```
- Then run the following command to preprocess the data using our scripts.
```
python3 ./preprocess/linker.py --data_folder_path path/to/Ohio_Data --extract_folder_path ./data
```

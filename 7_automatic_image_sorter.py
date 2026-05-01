from coastseg import classifier

# This script will automatically sort images in a directory using the model.
# The bad images are moved to a subdirectory called 'bad'.

# Select the RGB directory for an ROI from /data eg. CoastSeg/data/<ROI NAME>/jpg_files/preprocessed/RGB
# example: input_directory = r"C:\CoastSeg\data\ID_12_datetime06-05-23__04_16_45\jpg_files\preprocessed\RGB"
input_directory = r""

# run the classifier to automatically sort the images as good or bad based on the threshold (we recommend starting with a low threshold like 0.40 and then adjusting as needed)
classifier.sort_images_with_model(input_directory, threshold=0.40, print_status=True)

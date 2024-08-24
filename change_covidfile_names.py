import os

# Specify the directory containing the files
folder_path = 'COVID-19/csse_covid_19_daily_reports_us/'
# new_folder_path = 'COVID-19/csse_covid_19_daily_reports_us_new/'

cont = 0

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Split the filename and the extension
    name, ext = os.path.splitext(filename)
    
    # Check if the filename matches the expected date format MM-DD-YYYY
    if len(name) == 10 and name[2] == '-' and name[5] == '-':
        # Extract the month, day, and year
        month, day, year = name.split('-')
        
        # Create the new filename in the format YYYY-MM-DD
        new_name = f'{year}-{month}-{day}{ext}'
        
        # Build the full old and new file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
        
        print(f'Renamed: {filename} -> {new_name}')
        cont +=1
    else:
        print('SKIPPING FILE:', name)

print(cont, 'files renamed')
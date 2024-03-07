import os
import pandas as pd

def list_files_in_directory(directory_path):
    df_temp = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            crop_illness = root.split("\\")[-1]
            if root.split("\\")[-1] != "Invalid":
                df_temp.append(pd.DataFrame({
                            f'crop': crop_illness.split("___")[1],
                            f'illness': crop_illness.split("___")[-1],
                            f'crop_illness': crop_illness,
                        }, index=[0]))

    df_temp = pd.concat(df_temp, axis=0, ignore_index=True)
    df_temp.fillna(0, inplace=True)
    crop_dummies = pd.get_dummies(df_temp['crop'], prefix='crop', dtype=int)
    illness_dummies = pd.get_dummies(df_temp['illness'], prefix='illness', dtype=int)
    crop_illness_dummies = pd.get_dummies(df_temp['crop_illness'], prefix='crop_illness', dtype=int)
    df_temp = pd.concat([crop_dummies, illness_dummies, crop_illness_dummies, df_temp], axis=1)
    df_temp = df_temp.drop(columns=['crop', 'illness', 'crop_illness'])
    df_temp.to_csv("Dataset.csv", index=False)

    print(df_temp)


directory_path = 'Dataset'

if os.path.exists(directory_path):
    list_files_in_directory(directory_path)
else:
    print(f"The directory '{directory_path}' does not exist.")
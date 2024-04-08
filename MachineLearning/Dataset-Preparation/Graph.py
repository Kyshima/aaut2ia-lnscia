import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/Diana/Documents/aaut2ia-lnscia/MachineLearning/DatasetBinary128.csv")

# Count the occurrences of each unique combination of crop and illness
illness_counts = dataset.groupby(['crop', 'illness']).size().reset_index(name='count')

# Plot the histogram of illness instances
plt.bar(illness_counts['illness'], illness_counts['count'], color='skyblue', edgecolor='black')
plt.xlabel('Illness')
plt.ylabel('Frequency')
plt.title('Distribution of Illness Instances')
plt.xlim(-1, 14)  # Set x-axis limits from 0 to 14
plt.ylim(0, 600)  # Set y-axis limits from 0 to 5000
plt.grid(True)
plt.show()

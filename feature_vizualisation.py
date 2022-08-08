import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
from data_processing import *

df = pd.read_csv("wine_dataset_1/winemag-data-130k-v2.csv")

df_reviews = pd.read_pickle("df_reviews.pkl")
df_wines = pd.read_pickle("df_wines.pkl")
df.drop(df.columns[[0]], axis=1, inplace=True)

col_names = df_wines.columns
na_nb = [df_wines[col].isna().sum() for col in col_names]
per_na_nb = [item/21605*100 for item in na_nb]
per_na_nb, col_names = (list(t) for t in zip(*sorted(zip(per_na_nb, col_names), reverse=True)))
unique_nb = [df_wines[col].nunique() for col in col_names if col != 'varieties' and col != 'local' and col != 'use']
#unique_nb, col_names = (list(t) for t in zip(*sorted(zip(unique_nb, col_names), reverse=True)))
taster_values = df['taster_name'].value_counts()

# plt.hist(df_reviews['points'], bins = 20)
# plt.xlabel("Points")
# plt.title("Distribution of points")

# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))
# Horizontal Bar Plot
ax.barh(col_names, per_na_nb)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
	ax.spines[s].set_visible(False)
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
# Add x, y gridlines
ax.grid(b = True, color ='grey',
		linestyle ='-.', linewidth = 0.5,
		alpha = 0.2)
# Show top values
ax.invert_yaxis()
# Add annotation to bars
for i in ax.patches:
	plt.text(i.get_width()+0.2, i.get_y()+0.5,
			str(round((i.get_width()), 2)),
			fontsize = 10, fontweight ='bold',
			color ='grey')
# Add Plot Title
ax.set_title('Percentage of missing values for each feature',
			loc ='left', )
# Show Plot
plt.show()

print('Breakpoint')
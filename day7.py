import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("Spotify Dataset.csv")

# Drop unnecessary columns
data.drop(columns=['Unnamed: 0', 'Url_spotify', 'Uri', 'Url_youtube'], inplace=True)

# Fill missing values in Likes & Comments with 0
data['Likes'] = data['Likes'].fillna(0)
data['Comments'] = data['Comments'].fillna(0)

# Drop remaining rows with null values
data.dropna(inplace=True)

print(data.info())

#-----------------------------------#
# Q1: Top 10 Artists with Highest YouTube Views
#-----------------------------------#
artist_grouped = data.groupby('Artist')['Views'].sum()
artist_sorted = artist_grouped.sort_values(ascending=False)
print("\nTop 10 Artists by Views:\n", artist_sorted.head(10))

#-----------------------------------#
# Q2: Top 10 Tracks with Highest Spotify Streams
#-----------------------------------#
x = data[['Track', 'Stream']]
most_stream_track = x.sort_values(by='Stream', ascending=False).head(10)
print("\nTop 10 Tracks by Streams:\n", most_stream_track)

#-----------------------------------#
# Q3: Most Common Album Types
#-----------------------------------#
album_types = data['Album_type'].value_counts()
print("\nAlbum Types Count:\n", album_types)

plt.pie(album_types, labels=album_types.index, autopct='%1.1f%%',
        startangle=60, shadow=True, explode=(0.05, 0.05, 0.05))
plt.title("Album Type Distribution")
plt.show()

#-----------------------------------#
# Q4: Avg Views, Likes, Comments by Album Type
#-----------------------------------#
df = data.groupby('Album_type')[['Likes', 'Views', 'Comments']].mean().reset_index()
df_melted = df.melt(id_vars='Album_type', var_name='Attribute', value_name='Total')

sns.barplot(x='Album_type', y='Total', hue='Attribute', data=df_melted)
plt.title("Avg Views, Likes, Comments by Album Type")
plt.show()

#-----------------------------------#
# Q5: Top 5 YouTube Channels by Views
#-----------------------------------#
c_views = data.groupby('Channel')['Views'].sum().sort_values(ascending=False).head(5).reset_index()

sns.barplot(x='Views', y='Channel', data=c_views)
plt.title("Top 5 YouTube Channels by Views")
plt.show()

#-----------------------------------#
# Q6: Top Track by Views
#-----------------------------------#
top_track = data.sort_values(by='Views', ascending=False).head(1)
print("\nTop Track by Views:\n", top_track[['Track', 'Views']])

#-----------------------------------#
# Q7: Top 7 Tracks by Like-to-View Ratio
#-----------------------------------#
track_lv = data[data['Views'] > 0].copy()
track_lv['LV_Ratio'] = (track_lv['Likes'] / track_lv['Views']) * 100

top_lv = track_lv.sort_values(by='LV_Ratio', ascending=False).head(7)
print("\nTop 7 Tracks by Like-to-View Ratio:\n", top_lv[['Track', 'LV_Ratio']])

# Q7.A Lowest ratios
low_lv = track_lv.sort_values(by='LV_Ratio', ascending=True).head(3)
print("\nLowest 3 Tracks by Like-to-View Ratio:\n", low_lv[['Track', 'LV_Ratio']])

#-----------------------------------#
# Q8: Albums with Maximum Danceability
#-----------------------------------#
album_dance = data.groupby('Album')['Danceability'].sum().sort_values(ascending=False)
print("\nTop Albums by Danceability:\n", album_dance.head(10))

#-----------------------------------#
# Q9: Correlation between Views, Likes, Comments, Stream
#-----------------------------------#
df_corr = data[['Views', 'Likes', 'Comments', 'Stream']]
print("\nCorrelation Matrix:\n", df_corr.corr())

sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

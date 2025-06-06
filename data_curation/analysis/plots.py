import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# Load the dataframe.
df_mc_all = pd.read_csv("MC_ALL.csv")

# --- Data Preprocessing ---
# Convert 'publish_date' to datetime objects
df_mc_all['publish_date'] = pd.to_datetime(df_mc_all['publish_date'], errors='coerce')

# Drop rows where date conversion might have failed
df_mc_all.dropna(subset=['publish_date'], inplace=True)

# Define key columns for convenience
hyperpartisan_col = 'hyperpartisan_gold_label'
prct_col = 'prct_gold_label'
stance_col = 'stance_gold_label'
media_col = 'media_name'
date_col = 'publish_date'

# Create a directory for saving plots if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Set a general style for plots - optimized for small subfigures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")


# --- Plot 2: Proportion of Hyperpartisan Gold Labels per Top N media_name - NO TITLE ---
print("\n--- Generating plot for Hyperpartisan Gold Label per Top Media (No Title) ---")
N_media = 10  # Show top 10 media sources
top_media_sources_counts = df_mc_all[media_col].value_counts().nlargest(N_media)
top_media_list = top_media_sources_counts.index.tolist()
df_top_media = df_mc_all[df_mc_all[media_col].isin(top_media_list)].copy()

# Ensure 'hyperpartisan_gold_label' is treated as categorical for grouping
df_top_media.loc[:, hyperpartisan_col] = df_top_media[hyperpartisan_col].astype('category')

# Optimized for subfigure
plt.figure(figsize=(8, 6))  # Reduced from (15, 10)
hyperpartisan_by_media = df_top_media.groupby(media_col)[hyperpartisan_col].value_counts(normalize=True).mul(100).unstack(fill_value=0)
# Order by the original top_media_sources_counts order
hyperpartisan_by_media = hyperpartisan_by_media.loc[top_media_list]

hyperpartisan_by_media.plot(kind='barh', stacked=True, colormap='viridis', ax=plt.gca())
plt.xlabel('Percentage (%)', fontsize=14)  # Reduced from 24, shortened label
plt.ylabel('Media Outlet', fontsize=14)  # Reduced from 28, shortened label
plt.xticks(fontsize=11)  # Reduced from 20
plt.yticks(fontsize=20)  # Reduced from 22, smaller for media names
plt.legend(title='HP', prop={'size': 25}, title_fontsize=25, 
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Multi-line legend title
plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plot_filename = f"plots/hyperpartisan_gold_by_top_{N_media}_media_no_title.pdf"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_filename}")


# 2. Correlation between stance_gold_label and hyperpartisan_gold_label
print("\n--- Generating plot for Distribution of Hyperpartisan Gold Label by Stance ---")
df_mc_all[stance_col] = df_mc_all[stance_col].astype('category')

# Calculate proportions
stance_vs_hp_proportions = df_mc_all.groupby(stance_col)[hyperpartisan_col].value_counts(normalize=True).mul(100).unstack(fill_value=0)
stance_vs_hp_proportions.sort_index(inplace=True)

# Optimized for subfigure
plt.figure(figsize=(6, 5))  # Reduced from (10, 7)
ax = stance_vs_hp_proportions.plot(kind='bar', stacked=True, colormap='viridis', 
                                  ax=plt.gca(), width=0.7)  # Slightly thinner bars
plt.xlabel('Stance', fontsize=14)  # Reduced from 20, shortened label
plt.ylabel('Percentage (%)', fontsize=14)  # Reduced from 28, shortened label
plt.xticks(fontsize=12, rotation=0)  # Reduced from 18
plt.yticks(fontsize=12)  # Reduced from 22
plt.legend(title='HP', prop={'size': 20}, title_fontsize=20, 
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Multi-line legend title
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plot_filename = "plots/proportion_hyperpartisan_by_stance.pdf"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_filename}")

# 3. Correlation between stance_gold_label and prct_gold_label
print("\n--- Generating plot for Distribution of PRCT Gold Label by Stance ---")
# Calculate proportions
stance_vs_prct_proportions = df_mc_all.groupby(stance_col)[prct_col].value_counts(normalize=True).mul(100).unstack(fill_value=0)
stance_vs_prct_proportions.sort_index(inplace=True)

# Optimized for subfigure
plt.figure(figsize=(6, 5))  # Reduced from (10, 7)
stance_vs_prct_proportions.plot(kind='bar', stacked=True, colormap='coolwarm', 
                               ax=plt.gca(), width=0.7)  # Slightly thinner bars
plt.xlabel('Stance', fontsize=14)  # Reduced from 20, shortened label
plt.ylabel('Percentage (%)', fontsize=14)  # Reduced from 28, shortened label
plt.xticks(fontsize=12, rotation=0)  # Reduced from 18
plt.yticks(fontsize=12)  # Reduced from 22
plt.legend(title='PRCT', prop={'size': 20}, title_fontsize=20, 
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Reduced legend size, shortened title
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plot_filename = "plots/proportion_prct_by_stance.pdf"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_filename}")

print("\nOptimized plots for LaTeX subfigures have been generated and saved in the 'plots' directory.")
print("Key optimizations made:")
print("- Reduced figure sizes from large standalone plots to subfigure-appropriate dimensions")
print("- Scaled down font sizes while maintaining readability")
print("- Used multi-line legend titles and axis labels with \\n for better space utilization")
print("- Shortened axis labels to prevent overcrowding")
print("- Reduced legend sizes and simplified titles")
print("- Added high DPI (300) and bbox_inches='tight' for crisp PDF output")
print("- Adjusted bar widths and marker sizes for better visibility at small scale")
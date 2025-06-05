import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np

def load_and_analyze_csvs(path_to_dir, suffix="_merged.csv"):
    filenames = [os.path.join(path_to_dir, filename) 
                 for filename in os.listdir(path_to_dir) 
                 if filename.endswith(suffix)]
    dfs = [pd.read_csv(filename, encoding='utf-8') for filename in filenames]
    for filename, df in zip(filenames, dfs):
        print(f"{filename}: {df.shape[0]} rows")
    df_total = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return df_total

# Load data
data = load_and_analyze_csvs('/home/michele/Documenti/HYBRIDS/Experiment/Dataset_creation_MC/Final_dataset/data')
data.to_csv('MC_ALL.csv')

# Basic stats
ids = data['id'].nunique()
data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))  # Handle missing text
word_count_mean = data['word_count'].mean()
media = data['media_name'].nunique()
print("MEDIA", media)

# Calculate word count statistics by language
word_count_by_language = {}
if 'language' in data.columns:
    for lang in data['language'].unique():
        lang_data = data[data['language'] == lang]
        word_count_by_language[lang] = {
            'mean': lang_data['word_count'].mean(),
            'median': lang_data['word_count'].median(),
            'std': lang_data['word_count'].std(),
            'count': len(lang_data)
        }

# Focus on gold_label and _majority columns
gold_label_cols = ['hyperpartisan_gold_label', 'prct_gold_label', 'stance_gold_label']
majority_cols = ['loadedLanguage_majority', 'appealToFear_majority', 'nameCalling_majority']

# Define label types for proper handling
binary_cols = ['hyperpartisan_gold_label', 'prct_gold_label']
categorical_cols = ['stance_gold_label']

# Label distributions overall
label_dist = {}
for col in gold_label_cols + majority_cols:
    if col in data.columns:
        label_count = data[col].value_counts(dropna=False)
        label_dist[col] = label_count.to_dict()

# Label distributions by language
label_dist_by_lang = {}
languages = data['language'].unique() if 'language' in data.columns else []

for lang in languages:
    lang_data = data[data['language'] == lang]
    label_dist_by_lang[lang] = {}
    
    for col in gold_label_cols + majority_cols:
        if col in lang_data.columns:
            label_count = lang_data[col].value_counts(dropna=False)
            label_dist_by_lang[lang][col] = label_count.to_dict()

# Visualize distributions
def plot_distributions(data, gold_cols, majority_cols, binary_cols):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Gold Labels and Majority Annotations', fontsize=16)
    
    # Gold labels
    for i, col in enumerate(gold_cols):
        if col in data.columns:
            ax = axes[0, i]
            
            # Handle binary vs categorical differently
            if col in binary_cols:
                # For binary columns, ensure proper ordering (0, 1)
                counts = data[col].value_counts().sort_index()
                labels = ['Non-' + col.split('_')[0].title(), col.split('_')[0].title()]
                colors = ['lightblue', 'darkblue']
                bars = ax.bar(range(len(counts)), counts.values, color=colors, alpha=0.7)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels([f'{labels[i]}\n({counts.index[i]})' for i in range(len(counts))])
            else:
                # For categorical columns (stance)
                counts = data[col].value_counts()
                colors = ['green', 'red', 'gray'] if 'stance' in col else ['skyblue'] * len(counts)
                bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)], alpha=0.7)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=0)
            
            ax.set_title(f'{col.replace("_", " ").title()}')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom')
    
    # Majority annotations
    for i, col in enumerate(majority_cols):
        if col in data.columns:
            ax = axes[1, i]
            counts = data[col].value_counts()
            
            # Assume majority columns are binary (0/1 or similar)
            if set(counts.index).issubset({0, 1, '0', '1', 'yes', 'no', True, False}):
                colors = ['lightcoral', 'darkred']
            else:
                colors = ['lightcoral'] * len(counts)
            
            bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)], alpha=0.7)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=0)
            ax.set_title(f'{col.replace("_", " ").title()}')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Word count visualization by language
def plot_word_count_by_language(data, word_count_by_language):
    if not word_count_by_language:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of mean word counts
    languages = list(word_count_by_language.keys())
    means = [word_count_by_language[lang]['mean'] for lang in languages]
    stds = [word_count_by_language[lang]['std'] for lang in languages]
    counts = [word_count_by_language[lang]['count'] for lang in languages]
    
    bars = ax1.bar(languages, means, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(languages)])
    ax1.set_title('Average Word Count by Language')
    ax1.set_ylabel('Average Word Count')
    ax1.set_xlabel('Language')
    
    # Add value labels on bars
    for i, (bar, mean, count) in enumerate(zip(bars, means, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + bar.get_height()*0.01,
                f'{mean:.1f}\n(n={count})', ha='center', va='bottom', fontsize=10)
    
    # Box plot of word count distributions
    word_counts_by_lang = []
    lang_labels = []
    for lang in languages:
        lang_data = data[data['language'] == lang]['word_count']
        word_counts_by_lang.append(lang_data)
        lang_labels.append(f'{lang}\n(n={len(lang_data)})')
    
    ax2.boxplot(word_counts_by_lang, labels=lang_labels)
    ax2.set_title('Word Count Distribution by Language')
    ax2.set_ylabel('Word Count')
    ax2.set_xlabel('Language')
    
    plt.tight_layout()
    plt.show()

# Correlation analysis
def analyze_correlations(data, gold_cols, majority_cols, binary_cols):
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Create a subset with relevant columns
    relevant_cols = [col for col in gold_cols + majority_cols if col in data.columns]
    corr_data = data[relevant_cols].copy()
    
    # For correlation analysis, we need to handle different data types
    corr_data_encoded = corr_data.copy()
    
    # Binary columns can be used as-is (0,1)
    for col in binary_cols:
        if col in corr_data.columns:
            # Ensure they are numeric
            corr_data_encoded[col] = pd.to_numeric(corr_data_encoded[col], errors='coerce')
    
    # Categorical columns need label encoding
    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    
    for col in relevant_cols:
        if col not in binary_cols and col in corr_data.columns:
            le = LabelEncoder()
            mask = corr_data[col].notna()
            if mask.sum() > 0:
                corr_data_encoded.loc[mask, col] = le.fit_transform(corr_data[col][mask])
                le_dict[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Handle majority columns (assume they might be binary or need encoding)
    for col in majority_cols:
        if col in corr_data.columns and col not in le_dict:
            unique_vals = corr_data[col].dropna().unique()
            if len(unique_vals) <= 10:  # Reasonable number for encoding
                le = LabelEncoder()
                mask = corr_data[col].notna()
                if mask.sum() > 0:
                    corr_data_encoded.loc[mask, col] = le.fit_transform(corr_data[col][mask])
                    le_dict[col] = le
                    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Calculate correlation matrix
    correlation_matrix = corr_data_encoded.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, mask=mask, cbar_kws={"shrink": .8}, fmt='.3f')
    plt.title('Correlation Matrix: Gold Labels vs Majority Annotations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print correlation values for gold labels vs majority annotations
    print("\nKey Correlations (Gold Labels vs Majority Annotations):")
    for gold_col in gold_cols:
        if gold_col in correlation_matrix.columns:
            print(f"\n{gold_col}:")
            for maj_col in majority_cols:
                if maj_col in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[gold_col, maj_col]
                    strength = "Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak"
                    print(f"  vs {maj_col}: {corr_val:.3f} ({strength})")
    
    return correlation_matrix, le_dict

# Chi-square tests for categorical associations
def chi_square_tests(data, gold_cols, majority_cols):
    print("\n" + "="*50)
    print("CHI-SQUARE TESTS FOR ASSOCIATIONS")
    print("="*50)
    
    results = {}
    
    for gold_col in gold_cols:
        if gold_col not in data.columns:
            continue
            
        for maj_col in majority_cols:
            if maj_col not in data.columns:
                continue
                
            # Create contingency table
            contingency_table = pd.crosstab(data[gold_col], data[maj_col], 
                                          margins=False, dropna=False)
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate Cramér's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            results[f"{gold_col} vs {maj_col}"] = {
                'chi2': chi2,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'contingency_table': contingency_table
            }
            
            print(f"\n{gold_col} vs {maj_col}:")
            print(f"  Chi-square: {chi2:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cramér's V: {cramers_v:.4f}")
            print(f"  Association: {'Significant' if p_value < 0.05 else 'Not significant'}")
            print("  Contingency Table:")
            print(contingency_table)
    
    return results

# Generate enhanced LaTeX table with language breakdown and word counts
def generate_enhanced_latex_table(document_count, avg_words, word_count_by_language, label_dist, label_dist_by_lang, binary_cols):
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Dataset Statistics}")
    latex.append("\\begin{tabular}{lr}")
    latex.append("\\hline")
    latex.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    latex.append("\\hline")
    latex.append(f"Number of Documents & {document_count} \\\\")
    latex.append(f"Avg. Words per Text (Overall) & {avg_words:.2f} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Word count statistics by language
    if word_count_by_language:
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Word Count Statistics by Language}")
        latex.append("\\begin{tabular}{lrrrr}")
        latex.append("\\hline")
        latex.append("\\textbf{Language} & \\textbf{Count} & \\textbf{Mean} & \\textbf{Median} & \\textbf{Std Dev} \\\\")
        latex.append("\\hline")
        
        for lang, stats in word_count_by_language.items():
            latex.append(f"{lang} & {stats['count']} & {stats['mean']:.2f} & {stats['median']:.2f} & {stats['std']:.2f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")
    
    # Overall label distribution table
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Overall Label Distribution}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Label Type} & \\textbf{Class} & \\textbf{Count} & \\textbf{\\%} \\\\")
    latex.append("\\hline")
    
    for col, dist in label_dist.items():
        col_name = col.replace('_', '\\_')
        total = sum(dist.values())
        
        if col in binary_cols:
            # For binary columns, show meaningful labels
            base_name = col.split('_')[0].replace('prct', 'PRCT')
            for label, count in sorted(dist.items()):
                label_name = f"Non-{base_name}" if label == 0 else base_name
                percentage = (count / total) * 100
                latex.append(f"{col_name} & {label_name} ({label}) & {count} & {percentage:.1f}\\% \\\\")
        else:
            for label, count in dist.items():
                percentage = (count / total) * 100
                latex.append(f"{col_name} & {label} & {count} & {percentage:.1f}\\% \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Language-specific distribution tables
    if label_dist_by_lang:
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Label Distribution by Language}")
        
        # Create a comprehensive table with languages as columns
        languages = list(label_dist_by_lang.keys())
        lang_header = " & ".join([f"\\textbf{{{lang}}}" for lang in languages])
        
        latex.append(f"\\begin{{tabular}}{{l{'r' * len(languages)}}}")
        latex.append("\\hline")
        latex.append(f"\\textbf{{Label}} & {lang_header} \\\\")
        latex.append("\\hline")
        
        # For each label type and class, show counts across languages
        for col in gold_label_cols + majority_cols:
            if col in label_dist:
                col_name = col.replace('_', '\\_')
                latex.append(f"\\multicolumn{{{len(languages)+1}}}{{c}}{{\\textit{{{col_name}}}}} \\\\")
                
                # Get all possible labels for this column
                all_labels = set()
                for lang in languages:
                    if col in label_dist_by_lang[lang]:
                        all_labels.update(label_dist_by_lang[lang][col].keys())
                
                for label in sorted(all_labels):
                    if col in binary_cols:
                        base_name = col.split('_')[0].replace('prct', 'PRCT')
                        label_name = f"Non-{base_name}" if label == 0 else base_name
                        display_label = f"{label_name} ({label})"
                    else:
                        display_label = str(label)
                    
                    counts = []
                    for lang in languages:
                        count = label_dist_by_lang[lang].get(col, {}).get(label, 0)
                        counts.append(str(count))
                    
                    latex.append(f"{display_label} & {' & '.join(counts)} \\\\")
                
                latex.append("\\hline")
        
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
    
    return "\n".join(latex)

# Analyze distributions by language
def analyze_by_language(data, gold_cols, majority_cols, binary_cols):
    print("\n" + "="*60)
    print("LABEL DISTRIBUTIONS BY LANGUAGE") 
    print("="*60)
    
    if 'language' not in data.columns:
        print("No 'language' column found in data")
        return
    
    languages = data['language'].value_counts()
    print(f"\nLanguages in dataset:")
    for lang, count in languages.items():
        percentage = (count / len(data)) * 100
        print(f"  {lang}: {count} documents ({percentage:.1f}%)")
    
    # Detailed breakdown by language
    for lang in languages.index:
        lang_data = data[data['language'] == lang]
        print(f"\n{'-'*40}")
        print(f"LANGUAGE: {lang} ({len(lang_data)} documents)")
        print(f"{'-'*40}")
        
        for col in gold_cols + majority_cols:
            if col in lang_data.columns:
                print(f"\n{col}:")
                dist = lang_data[col].value_counts(dropna=False)
                total = len(lang_data)
                
                if col in binary_cols:
                    col_name = col.split('_')[0]
                    for label, count in sorted(dist.items()):
                        label_name = f"Non-{col_name}" if label == 0 else col_name.title()
                        percentage = (count / total) * 100
                        print(f"  {label_name} ({label}): {count} ({percentage:.1f}%)")
                else:
                    for label, count in dist.items():
                        percentage = (count / total) * 100
                        print(f"  {label}: {count} ({percentage:.1f}%)")

# Cross-language correlation analysis
def cross_language_analysis(data, gold_cols, majority_cols):
    print("\n" + "="*60)
    print("CROSS-LANGUAGE LABEL CONSISTENCY")
    print("="*60)
    
    if 'language' not in data.columns:
        return
    
    languages = data['language'].unique()
    
    for col in gold_cols + majority_cols:
        if col in data.columns:
            print(f"\n{col}:")
            
            # Calculate distribution for each language
            lang_distributions = {}
            for lang in languages:
                lang_data = data[data['language'] == lang]
                if len(lang_data) > 0:
                    dist = lang_data[col].value_counts(normalize=True).sort_index()
                    lang_distributions[lang] = dist
            
            # Show proportional differences
            if len(lang_distributions) > 1:
                print("  Proportional distributions:")
                all_labels = set()
                for dist in lang_distributions.values():
                    all_labels.update(dist.index)
                
                for label in sorted(all_labels):
                    proportions = []
                    for lang in languages:
                        if lang in lang_distributions:
                            prop = lang_distributions[lang].get(label, 0)
                            proportions.append(f"{lang}: {prop:.3f}")
                    print(f"    {label}: {', '.join(proportions)}")

# Execute analysis
print("DATASET OVERVIEW")
print("="*50)
print(f"Total documents: {ids}")
print(f"Average words per text (overall): {word_count_mean:.2f}")

# Display word count statistics by language
if word_count_by_language:
    print("\nWORD COUNT STATISTICS BY LANGUAGE")
    print("="*50)
    for lang, stats in word_count_by_language.items():
        print(f"{lang}:")
        print(f"  Documents: {stats['count']}")
        print(f"  Mean words: {stats['mean']:.2f}")
        print(f"  Median words: {stats['median']:.2f}")
        print(f"  Std deviation: {stats['std']:.2f}")

print("\nOVERALL LABEL DISTRIBUTIONS")
print("="*50)
for col, dist in label_dist.items():
    print(f"\n{col}:")
    total = sum(dist.values())
    
    if col in binary_cols:
        # For binary columns, show meaningful labels
        col_name = col.split('_')[0]
        for label, count in sorted(dist.items()):
            label_name = f"Non-{col_name}" if label == 0 else col_name.title()
            percentage = (count / total) * 100
            print(f"  {label_name} ({label}): {count} ({percentage:.1f}%)")
    else:
        # For categorical columns
        for label, count in dist.items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

# Analyze by language
analyze_by_language(data, gold_label_cols, majority_cols, binary_cols)

# Cross-language analysis
cross_language_analysis(data, gold_label_cols, majority_cols)

# Plot word count distributions by language
plot_word_count_by_language(data, word_count_by_language)

# Plot distributions (with all required arguments)
plot_distributions(data, gold_label_cols, majority_cols, binary_cols)

# Analyze correlations
correlation_matrix, label_encoders = analyze_correlations(data, gold_label_cols, majority_cols, binary_cols)

# Chi-square tests
chi_square_results = chi_square_tests(data, gold_label_cols, majority_cols)

# Generate LaTeX table with language breakdown and word counts
latex_table = generate_enhanced_latex_table(ids, word_count_mean, word_count_by_language, label_dist, label_dist_by_lang, binary_cols)
print("\n" + "="*50)
print("LATEX TABLES")
print("="*50)
print(latex_table)
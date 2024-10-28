import os
import re
import json
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

nltk.download('punkt')
OUTPUT_DIR = 'output_data'
REFERENCE_DIR = 'data'
NUM_FILES = 100
positive_match_scores = []
f1_section_scores = []
f1_anatomical_scores = []


def load_section_titles():
    """Load section titles from a JSON file."""
    with open('section_titles.json', 'r', encoding='utf-8') as f:
        return json.load(f)

section_titles = load_section_titles()


def split_text(text):
    """Clean the content by removing specific tags."""
    section_order = ['רקע', 'טכניקה', 'ממצאים', 'סיכום']
    content = re.sub(r'<ACCNUMBER>.*?</ACCNUMBER>', '', text, flags=re.DOTALL)
    content = re.sub(r'<RESULTTEXT>', '', content)
    content = re.sub(r'</RESULTTEXT>', '', content)
    split_pattern = r'(' + '|'.join(re.escape(title) for titles in section_titles.values() for title in titles) + r')'
    chunks = re.split(split_pattern, content)
    sections = {section_name: '' for section_name in section_order}
    current_section_index = 0
    i = 0
    while i < len(chunks):
        chunk = chunks[i].strip()
        if chunk in section_titles:
            section_name = next((key for key, values in section_titles.items() if chunk in values), None)
            current_section_index = section_order.index(section_name)
            i += 1
            if i < len(chunks):
                sections[section_name] += chunks[i].strip() + '\n'
        else:
            section_name = section_order[current_section_index]
            sections[section_name] += chunk + '\n'
        i += 1
    findings_start = [i for i in [content.find(title) for title in section_titles['טכניקה']] if i != -1]
    if len(findings_start) == 0:
        findings_start = [i for i in [content.find(title) for title in section_titles['רקע']] if i != -1]
    findings_end = [i for i in [content.find(title) for title in section_titles['סיכום']] if i != -1]
    if len(findings_start) != 0 and len(findings_end) != 0:
        findings_content = content[findings_start[0]:findings_end[0]].strip()
        findings_paragraphs = re.split(r'\n{3,}', findings_content)
        if len(findings_paragraphs) > 1:
            sections['ממצאים'] = findings_paragraphs[1:]
        new_findings_content = []
        for paragraph in sections['ממצאים']:
            splitted_paragraph = paragraph.split('\n')
            new_findings_content.append(splitted_paragraph)
        sections['ממצאים'] = new_findings_content
    return sections


def check_positive_match(output_sections, reference_sections):
    """Check if the output sections positively match the reference sections."""
    for section in ['רקע', 'טכניקה', 'סיכום']:
        if set(output_sections.get(section)) != set(reference_sections.get(section)):
            return False
    output_findings_section = output_sections.get('ממצאים')
    reference_findings_section = reference_sections.get('ממצאים')
    if len(output_findings_section) != len(reference_findings_section):
        return False
    for i in range(len(reference_findings_section)):
        if set(output_findings_section[i]) != set(reference_findings_section[i]):
            return False
    return True


def calculate_f1_scores(output_sections, reference_sections):
    """Calculate F1 scores for structuring and anatomical labeling tasks."""
    output_labels_structuring = []
    reference_labels_structuring = []
    for section in ['רקע', 'טכניקה', 'ממצאים', 'סיכום']:
        if section != 'ממצאים':
            output_labels_structuring.extend(output_sections[section].split())
            reference_labels_structuring.extend(reference_sections[section].split())
        else:
            output_labels_structuring.extend(['ממצאים'] * len(output_sections['ממצאים']))
            reference_labels_structuring.extend(['ממצאים'] * len(reference_sections['ממצאים']))
    f1_structuring = f1_score(reference_labels_structuring, output_labels_structuring, average='macro')
    output_labels_anatomical = []
    reference_labels_anatomical = []
    for idx in range(len(reference_sections['ממצאים'])):
        output_labels_anatomical.extend([f'ממצאים - {label}' for label in output_sections['ממצאים'][idx]])
        reference_labels_anatomical.extend([f'ממצאים - {label}' for label in reference_sections['ממצאים'][idx]])
    f1_anatomical = f1_score(reference_labels_anatomical, output_labels_anatomical, average='macro')
    return f1_structuring, f1_anatomical


def evaluate_model_for_files():
    """Evaluate model for multiple files and collect metrics."""
    for i in range(1, NUM_FILES + 1):
        output_file = os.path.join(OUTPUT_DIR, f"sample_{i}.txt")
        reference_file = os.path.join(REFERENCE_DIR, f"sample_{i}.txt")
        output_text = read_file_content(output_file)
        reference_text = read_file_content(reference_file)
        output_sections = split_text(output_text)
        reference_sections = split_text(reference_text)
        pm = check_positive_match(output_sections, reference_sections)
        positive_match_scores.append(pm)
        f1_section, f1_anatomical = calculate_f1_scores(output_sections, reference_sections)
        f1_section_scores.append(f1_section)
        f1_anatomical_scores.append(f1_anatomical)


def read_file_content(file_path):
    """Read file content with appropriate encoding."""
    try:
        with open(file_path, 'r', encoding='windows-1255') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    return content


def create_df():
    """Create a DataFrame from the evaluation metrics."""
    data = {
        'File': list(range(1, NUM_FILES + 1)),
        'Positive Match': positive_match_scores,
        'F1 Score - Section Structuring': f1_section_scores,
        'F1 Score - Anatomical Grouping': f1_anatomical_scores
    }
    return pd.DataFrame(data)


def calculate_avg_metrics(df):
    """Calculate average metrics from the DataFrame."""
    avg_metrics = df.mean(numeric_only=True)
    accuracy = df['Positive Match'].sum() / NUM_FILES
    avg_metrics['Accuracy'] = accuracy * 100
    return avg_metrics


def output_average_metrics_table(avg_metrics):
    """Output a table for Average Metrics."""
    table = pd.DataFrame(avg_metrics).reset_index()
    table.columns = ['Metric', 'Average Score']
    print("\nAverage Metrics Table:")
    print(table.to_string(index=False))


def plot_metrics(df):
    """Plot Positive Match and F1 Score metrics."""
    # Plot Positive Match
    plt.figure(figsize=(8, 6))
    match_counts = df['Positive Match'].value_counts().sort_index()
    match_labels = ['False', 'True']
    counts = [match_counts.get(False, 0), match_counts.get(True, 0)]
    plt.bar(match_labels, counts, color=['red', 'green'])
    plt.title('Positive Match Counts')
    plt.xlabel('Positive Match')
    plt.ylabel('Number of Files')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot F1 Scores for Structuring and Anatomical Grouping
    plt.figure(figsize=(10, 6))
    plt.plot(df['File'], df['F1 Score - Section Structuring'], marker='o', color='blue', label='F1 Score - Section Structuring')
    plt.plot(df['File'], df['F1 Score - Anatomical Grouping'], marker='o', color='orange', label='F1 Score - Anatomical Grouping')
    plt.title('F1 Scores per File')
    plt.xlabel('File Number')
    plt.ylabel('F1 Score')
    plt.xticks(df['File'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_results_to_csv(df, filename='evaluation_results.csv'):
    """Save evaluation results to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    evaluate_model_for_files()
    df = create_df()
    avg_metrics = calculate_avg_metrics(df)
    output_average_metrics_table(avg_metrics)
    plot_metrics(df)
    save_results_to_csv(df)

if __name__ == '__main__':
    main()

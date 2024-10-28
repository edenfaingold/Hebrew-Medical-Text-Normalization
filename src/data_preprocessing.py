import os
import re
import json


def load_head_regions():
    """Load head regions from a JSON file."""
    with open('hierarchical_head_regions.json', 'r', encoding='utf-8') as f:
        head_regions = json.load(f)
    return head_regions


def create_subregion_to_main_mapping(head_regions):
    """Create a mapping from subregions to main regions."""
    subregion_to_main = {}
    for main_region, sub_regions in head_regions.items():
        for sub_region in sub_regions:
            subregion_to_main[sub_region] = main_region
    return subregion_to_main


def load_section_titles():
    """Load section titles from a JSON file."""
    with open('section_titles.json', 'r', encoding='utf-8') as f:
        section_titles = json.load(f)
    return section_titles


def get_paragraphs(section):
    """Split a section into paragraphs."""
    paragraphs = re.split(r'\n{2,}', section)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            yield paragraph


def find_main_head_regions(sentence, subregion_to_main, head_regions_sections):
    """Find the main head regions in a sentence."""
    found_main_regions = set()
    for sub_region, main_region in subregion_to_main.items():
        if sub_region in sentence:
            found_main_regions.add(main_region)
    if not found_main_regions:
        if len(head_regions_sections) == 0:
            return 'ראש'
        max_region = max(head_regions_sections.items(), key=lambda x: x[1])[0]
        return max_region
    if len(found_main_regions) > 1:
        max_region = max(found_main_regions, key=lambda x: head_regions_sections.get(x, float('-inf')))
        if not max_region:
            main_region = next(iter(found_main_regions))
            head_regions_sections[main_region] = head_regions_sections.get(main_region, 0) + 1
            return main_region
        head_regions_sections[max_region] = head_regions_sections.get(max_region, 0) + 1
        return max_region
    main_region = next(iter(found_main_regions))
    head_regions_sections[main_region] = head_regions_sections.get(main_region, 0) + 1
    return main_region


def read_file_content(file_path):
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='windows-1255') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    return content


def clean_content(content):
    """Clean the content by removing specific tags."""
    content = re.sub(r'<ACCNUMBER>.*?</ACCNUMBER>', '', content, flags=re.DOTALL)
    content = re.sub(r'<RESULTTEXT>', '', content)
    content = re.sub(r'</RESULTTEXT>', '', content)
    return content


def create_section_titles_mappings():
    """Create mappings for section titles."""
    section_order = ['רקע', 'טכניקה', 'ממצאים', 'סיכום']
    section_titles = load_section_titles()
    title_to_section = {title: section for section, titles in section_titles.items() for title in titles}
    return section_order, section_titles, title_to_section


def split_content_into_sections(content, title_to_section, section_order):
    """Split content into sections based on the section titles."""
    split_pattern = r'(' + '|'.join(re.escape(title) for title in title_to_section.keys()) + r')'
    chunks = re.split(split_pattern, content)
    sections = {section_name: '' for section_name in section_order}
    current_section_index = 0
    i = 0
    while i < len(chunks):
        chunk = chunks[i].strip()
        if chunk in title_to_section:
            section_name = title_to_section[chunk]
            current_section_index = section_order.index(section_name)
            i += 1
            if i < len(chunks):
                sections[section_name] += chunks[i].strip() + '\n'
        else:
            section_name = section_order[current_section_index]
            sections[section_name] += chunk + '\n'
        i += 1
    technique_start = content.find('טכניקה:')
    summary_start = content.find('סיכום:')
    if technique_start != -1 and summary_start != -1:
        findings_content = content[technique_start:summary_start].strip()
        findings_paragraphs = re.split(r'\n{2,}', findings_content)
        if len(findings_paragraphs) > 1:
            sections['ממצאים'] = '\n\n'.join(
                findings_paragraphs[1:])
    return sections


def process_findings_section(section_content, subregion_to_main, head_regions_sections):
    """Process the findings section."""
    sentences = []
    labels = []
    paragraphs = get_paragraphs(section_content)
    for paragraph in paragraphs:
        paragraph_sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        for sentence in paragraph_sentences:
            sentence = sentence.strip()
            if sentence:
                found_main_region = find_main_head_regions(sentence, subregion_to_main, head_regions_sections)
                main_region = found_main_region
                composite_label = f'ממצאים - {main_region}'
                sentences.append(sentence)
                labels.append(composite_label)
    return sentences, labels


def process_other_section(section_content, section_name):
    """Process a section other than the findings section."""
    sentences = []
    labels = []
    section_sentences = re.split(r'(?<=[.!?])\s+', section_content)
    for sentence in section_sentences:
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
            labels.append(section_name)
    return sentences, labels


def process_sections(sections, section_order, subregion_to_main):
    """Process the sections of a file."""
    all_sentences = []
    all_labels = []
    head_regions_sections = {}
    for section_name in section_order:
        section_content = sections[section_name]
        if not section_content.strip():
            continue
        if section_name == 'ממצאים':
            sentences, labels = process_findings_section(section_content, subregion_to_main,
                                                         head_regions_sections)
        else:
            sentences, labels = process_other_section(section_content, section_name)
        all_sentences.extend(sentences)
        all_labels.extend(labels)
    return all_sentences, all_labels


def preprocess_file(file_path, subregion_to_main, title_to_section, section_order):
    """Preprocess a file."""
    content = read_file_content(file_path)
    content = clean_content(content)
    sections = split_content_into_sections(content, title_to_section, section_order)
    sentences, labels = process_sections(sections, section_order, subregion_to_main)
    return sentences, labels


def preprocess_data(data_directory):
    """Preprocess the data."""
    head_regions = load_head_regions()
    subregion_to_main = create_subregion_to_main_mapping(head_regions)
    section_order, _, title_to_section = create_section_titles_mappings()
    files = []
    all_sentences = []
    all_labels = []
    file_names = [f for f in os.listdir(data_directory) if f.endswith('.txt')]
    for file_name in file_names:
        file_path = os.path.join(data_directory, file_name)
        sentences, labels = preprocess_file(file_path, subregion_to_main, title_to_section, section_order)
        files.append(file_name)
        all_sentences.extend(sentences)
        all_labels.extend(labels)

    return files, all_sentences, all_labels


def create_label_mappings(labels):
    """Create label mappings."""
    label_list = sorted(list(set(labels)))
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_list, label_to_id, id_to_label


def encode_labels(labels, label_to_id):
    """Encode the labels."""
    return [label_to_id[label] for label in labels]

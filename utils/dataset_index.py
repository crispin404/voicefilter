import csv
import json
import os
import random
import re
from glob import glob


REQUIRED_DIRS = {
    'vowel_dir': '元音',
    'snore_dir': '鼾声',
    'mix_dir': '合成声_1',
}
OPTIONAL_DIR_KEYS = {'mix_dir'}

VOWEL_FILES = ['a1_1.wav', 'e1_1.wav', 'i1_1.wav', 'o1_1.wav', 'u1_1.wav']
SNORE_PATTERN = re.compile(r'^hs_(\d+)_([0-9]+)\.wav$', re.IGNORECASE)
MIX_PATTERN = re.compile(r'^hs_(\d+)_([a-zA-Z]+)_([0-9]+)\.wav$', re.IGNORECASE)

INFO_FIELD_PATTERNS = {
    'name': [r'姓名[:：]?\s*(.+)'],
    'gender': [r'性别[:：]?\s*(.+)'],
    'age': [r'年龄[:：]?\s*([0-9]+)'],
    'height_cm': [r'身高[:：]?\s*([0-9]+(?:\.[0-9]+)?)'],
    'weight_kg': [r'体重[:：]?\s*([0-9]+(?:\.[0-9]+)?)'],
    'neck_cm': [r'颈围[:：]?\s*([0-9]+(?:\.[0-9]+)?)'],
    'waist_cm': [r'腰围[:：]?\s*([0-9]+(?:\.[0-9]+)?)'],
}


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def normalize_path(path):
    return os.path.abspath(path).replace('\\', '/')


def list_wavs(path):
    return sorted(
        wav for wav in glob(os.path.join(path, '*.wav'))
        if os.path.isfile(wav)
    )


def list_wavs_if_dir(path):
    if not path or not os.path.isdir(path):
        return []
    return list_wavs(path)


def strip_wav_extension(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def subject_mix_dir(subject, mix_dir_name=None):
    if mix_dir_name:
        return normalize_path(os.path.join(subject['subject_dir'], mix_dir_name))
    return normalize_path(subject.get('mix_dir', os.path.join(subject['subject_dir'], REQUIRED_DIRS['mix_dir'])))


def list_subject_mix_paths(subject, mix_dir_name=None):
    return [normalize_path(path) for path in list_wavs_if_dir(subject_mix_dir(subject, mix_dir_name=mix_dir_name))]


def find_subject_dirs(data_root):
    return sorted(
        os.path.join(data_root, name)
        for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    )


def scan_subjects(data_root):
    subjects = []
    for class_index, subject_dir in enumerate(find_subject_dirs(data_root)):
        subject_name = os.path.basename(subject_dir)
        subject_id = sanitize_subject_id(subject_name)
        item = {
            'subject_id': subject_id,
            'class_index': class_index,
            'subject_name': subject_name,
            'subject_dir': normalize_path(subject_dir),
            'info_path': normalize_path(os.path.join(subject_dir, 'info.txt')),
        }
        missing = []
        optional_missing = []
        for key, dirname in REQUIRED_DIRS.items():
            path = os.path.join(subject_dir, dirname)
            item[key] = normalize_path(path)
            if not os.path.isdir(path):
                if key in OPTIONAL_DIR_KEYS:
                    optional_missing.append(dirname)
                else:
                    missing.append(dirname)
        if not os.path.isfile(os.path.join(subject_dir, 'info.txt')):
            missing.append('info.txt')

        item['exists'] = len(missing) == 0
        item['missing'] = missing
        item['optional_missing'] = optional_missing
        item['vowel_paths'] = [
            normalize_path(os.path.join(subject_dir, REQUIRED_DIRS['vowel_dir'], filename))
            for filename in VOWEL_FILES
        ]
        item['snore_paths'] = [normalize_path(path) for path in list_wavs(os.path.join(subject_dir, REQUIRED_DIRS['snore_dir']))]
        item['mix_paths'] = [normalize_path(path) for path in list_wavs(os.path.join(subject_dir, REQUIRED_DIRS['mix_dir']))]
        subjects.append(item)
    return subjects


def sanitize_subject_id(name):
    safe = re.sub(r'\s+', '_', name.strip())
    safe = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', safe)
    return safe.strip('_') or 'subject'


def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(rows, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_subjects(subjects_path):
    subjects = load_json(subjects_path)
    return [subject for subject in subjects if subject.get('exists', True)]


def load_csv_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def split_subjects(subjects, train_count, val_count, test_count, seed):
    total = train_count + val_count + test_count
    if len(subjects) < total:
        raise ValueError('Not enough subjects for requested split: %d < %d' % (len(subjects), total))

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    train_subjects = shuffled[:train_count]
    val_subjects = shuffled[train_count:train_count + val_count]
    test_subjects = shuffled[train_count + val_count:train_count + val_count + test_count]
    return {
        'train': train_subjects,
        'val': val_subjects,
        'test': test_subjects,
    }


def save_subject_ids(subjects, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        for subject in subjects:
            f.write(subject['subject_id'] + '\n')


def load_subject_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def parse_info_file(path):
    result = {
        'name': '',
        'gender': '',
        'age': '',
        'height_cm': '',
        'weight_kg': '',
        'neck_cm': '',
        'waist_cm': '',
        'bmi': '',
    }
    if not os.path.isfile(path):
        return result

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    text = '\n'.join(lines)

    for field, patterns in INFO_FIELD_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                result[field] = match.group(1).strip()
                break

    height = safe_float(result['height_cm'])
    weight = safe_float(result['weight_kg'])
    if height and weight:
        height_m = height / 100.0 if height > 3 else height
        if height_m > 0:
            result['bmi'] = round(weight / (height_m * height_m), 2)
    return result


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(rows, path, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_clean_index(snore_dir):
    mapping = {}
    for path in list_wavs(snore_dir):
        match = SNORE_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        inner_id = int(match.group(1))
        snore_index = int(match.group(2))
        mapping[(inner_id, snore_index)] = path
    return mapping


def parse_mix_filename(path):
    match = MIX_PATTERN.match(os.path.basename(path))
    if match is None:
        return None
    return {
        'inner_id': int(match.group(1)),
        'noise_type': match.group(2),
        'snore_index': int(match.group(3)),
    }


def build_manifest_rows(subjects, subject_ids, processed_root=None, mix_dir_name=None, snr_lookup=None):
    subject_id_set = set(subject_ids)
    selected = [subject for subject in subjects if subject['subject_id'] in subject_id_set]
    rows = []
    for subject in selected:
        rows.extend(
            build_subject_manifest_rows(
                subject,
                processed_root=processed_root,
                mix_dir_name=mix_dir_name,
                snr_lookup=snr_lookup,
            )
        )
    return rows


def build_subject_manifest_rows(subject, processed_root=None, mix_dir_name=None, snr_lookup=None):
    raw_vowel_dir = subject['vowel_dir']
    raw_snore_dir = subject['snore_dir']
    raw_mix_dir = subject_mix_dir(subject, mix_dir_name=mix_dir_name)

    clean_index = build_clean_index(raw_snore_dir)
    rows = []
    for mix_path in list_wavs_if_dir(raw_mix_dir):
        mix_meta = parse_mix_filename(mix_path)
        if mix_meta is None:
            continue

        clean_path = clean_index.get((mix_meta['inner_id'], mix_meta['snore_index']))
        if clean_path is None:
            continue

        snr_meta = None
        if snr_lookup is not None:
            snr_meta = snr_lookup.get((subject['subject_id'], os.path.basename(mix_path)))
            if snr_meta is None:
                snr_meta = snr_lookup.get(normalize_path(mix_path))

        row = {
            'subject_id': subject['subject_id'],
            'class_index': subject['class_index'],
            'clean_path': normalize_path(resolve_processed_clean_path(clean_path, mix_path, subject['subject_id'], processed_root)),
            'mix_path': normalize_path(resolve_processed_audio_path(mix_path, subject['subject_id'], 'mix', processed_root)),
            'noise_type': mix_meta['noise_type'],
            'snore_index': mix_meta['snore_index'],
            'vowel_paths': [
                normalize_path(resolve_processed_audio_path(os.path.join(raw_vowel_dir, filename), subject['subject_id'], 'vowel', processed_root))
                for filename in VOWEL_FILES
            ],
            'embedding_path': normalize_path(resolve_embedding_path(subject['subject_id'], processed_root)),
            'info_path': normalize_path(subject['info_path']),
        }
        if snr_meta:
            row.update(extract_snr_fields(snr_meta))
        rows.append(row)
    return rows


def resolve_processed_audio_path(raw_path, subject_id, kind, processed_root):
    if processed_root is None:
        return raw_path

    processed_dir = os.path.join(processed_root, kind, subject_id)
    filename = strip_wav_extension(raw_path) + '.wav'
    return os.path.join(processed_dir, filename)


def resolve_processed_clean_path(raw_clean_path, mix_path, subject_id, processed_root):
    if processed_root is None:
        return raw_clean_path

    clean_dir = os.path.join(processed_root, 'clean', subject_id)
    pair_filename = '%s_clean.wav' % strip_wav_extension(mix_path)
    pair_path = os.path.join(clean_dir, pair_filename)
    if os.path.isfile(pair_path):
        return pair_path

    return resolve_processed_audio_path(raw_clean_path, subject_id, 'clean', processed_root)


def build_snr_lookup(rows):
    lookup = {}
    for row in rows:
        subject_id = row.get('subject_id')
        mix_basename = ''
        if row.get('mix_file'):
            mix_basename = os.path.basename(row['mix_file'])
        elif row.get('output_mix_file'):
            mix_basename = os.path.basename(row['output_mix_file'])

        if subject_id and mix_basename:
            lookup[(subject_id, mix_basename)] = row

        for path_key in ['mix_file', 'output_mix_file']:
            path = row.get(path_key)
            if path:
                lookup[normalize_path(path)] = row
    return lookup


def extract_snr_fields(row):
    result = {}
    for key in ['target_snr_db', 'actual_snr_db', 'duration_seconds']:
        value = row.get(key)
        if value in (None, ''):
            continue
        parsed = safe_float(value)
        result[key] = parsed if parsed is not None else value
    warning = row.get('warning')
    if warning:
        result['snr_warning'] = warning
    return result


def resolve_embedding_path(subject_id, processed_root):
    if processed_root is None:
        return os.path.join('processed', 'embeddings', '%s.npy' % subject_id)
    return os.path.join(processed_root, 'embeddings', '%s.npy' % subject_id)

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_index import ensure_dir

try:
    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.font_manager import FontProperties, fontManager, findfont
except ModuleNotFoundError as exc:
    raise SystemExit(
        'matplotlib is required to run scripts/plot_evaluation_figures.py. '
        'Please install the project requirements first.'
    ) from exc


DEFAULT_EVAL_DIR = os.path.join('outputs', 'platformax', 'eval')
DEFAULT_OUTPUT_DIR = os.path.join('outputs', 'figures')
DEFAULT_SUBJECTS_JSON = os.path.join('metadata', 'subjects.json')
BASELINE_METRICS_NAME = 'metrics.csv'
DOUBLE_NOISE_METRICS_NAME = 'metrics_12.csv'
TRIPLE_NOISE_METRICS_NAME = 'metrics_13.csv'
REQUIRED_COLUMNS = {'subject_id', 'noise_type', 'si_sdr_improvement'}
SINGLE_NOISE_ORDER = ['nz', 'dpt', 'jb', 'ye', 'xs', 'km', 'xcq', 'qm']
SUBJECT_LABEL_MAP = {
    '张中兵': 'subject 1',
    '周长保': 'subject 2',
    '许爱林': 'subject 3',
    '周前进': 'subject 4',
    '刘晓峰': 'subject 5',
}
SUBJECT_LABEL_ORDER = ['subject 1', 'subject 2', 'subject 3', 'subject 4', 'subject 5']
NOISE_LABEL_MAP = {
    'dpt': 'sneeze',
    'jb': 'footstep',
    'nz': 'alarm',
    'km': 'dooropen',
    'qm': 'knock',
    'ye': 'babycry',
    'xs': 'laugh',
    'xcq': 'vacuum',
}
SCIENTIFIC_BLUE = '#5B84B1'
SCIENTIFIC_BLUE_DARK = '#2F5D8A'
GRID_COLOR = '#D8E1EA'
HEATMAP_CMAP = 'YlGnBu'
DEFAULT_FONT_FALLBACKS = [
    'DejaVu Sans',
    'Arial Unicode MS',
]
CJK_FONT_FAMILIES = [
    'Noto Sans CJK SC',
    'Source Han Sans SC',
    'Noto Serif CJK SC',
]
CJK_FONT_PATHS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot thesis-ready figures from Platformax evaluation CSV outputs'
    )
    parser.add_argument(
        '--eval-dir',
        default=DEFAULT_EVAL_DIR,
        help='Directory containing metrics.csv, metrics_12.csv, and metrics_13.csv',
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Directory used to save generated figure images',
    )
    parser.add_argument(
        '--metrics-csv',
        default=None,
        help='Path to the main metrics CSV, defaults to <eval-dir>/metrics.csv',
    )
    parser.add_argument(
        '--metrics-12-csv',
        default=None,
        help='Optional path to the double-noise metrics CSV, defaults to <eval-dir>/metrics_12.csv',
    )
    parser.add_argument(
        '--metrics-13-csv',
        default=None,
        help='Optional path to the triple-noise metrics CSV, defaults to <eval-dir>/metrics_13.csv',
    )
    parser.add_argument(
        '--subjects-json',
        default=DEFAULT_SUBJECTS_JSON,
        help='Optional subjects.json used to map subject_id to display names',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI for saved images',
    )
    return parser.parse_args()


def resolve_csv_path(cli_path, eval_dir, filename):
    if cli_path:
        return cli_path
    return os.path.join(eval_dir, filename)


def find_registered_font_path(font_family):
    try:
        font_path = findfont(font_family, fallback_to_default=False)
    except Exception:
        return None
    if font_path and os.path.isfile(font_path):
        return font_path
    return None


def family_names_for_path(font_path):
    normalized_target = os.path.normcase(os.path.abspath(font_path))
    family_names = []
    for font_entry in fontManager.ttflist:
        entry_path = os.path.normcase(os.path.abspath(font_entry.fname))
        if entry_path != normalized_target:
            continue
        if font_entry.name and font_entry.name not in family_names:
            family_names.append(font_entry.name)
    return family_names


def choose_best_family_name(family_names):
    for preferred_family in CJK_FONT_FAMILIES:
        if preferred_family in family_names:
            return preferred_family
    for family_name in family_names:
        if family_name.endswith(' SC') or ' CJK SC' in family_name:
            return family_name
    return family_names[0] if family_names else None


def resolve_cjk_font():
    for font_family in CJK_FONT_FAMILIES:
        font_path = find_registered_font_path(font_family)
        if not font_path:
            continue
        family_names = family_names_for_path(font_path)
        chosen_family = choose_best_family_name(family_names) or font_family
        return {
            'source': 'family',
            'requested_family': font_family,
            'path': font_path,
            'family': chosen_family,
        }

    for font_path in CJK_FONT_PATHS:
        if not os.path.isfile(font_path):
            continue
        try:
            fontManager.addfont(font_path)
        except Exception:
            continue
        family_names = family_names_for_path(font_path)
        chosen_family = choose_best_family_name(family_names)
        if chosen_family:
            return {
                'source': 'path',
                'requested_family': chosen_family,
                'path': font_path,
                'family': chosen_family,
            }
        try:
            fallback_family = FontProperties(fname=font_path).get_name()
        except Exception:
            fallback_family = None
        if fallback_family:
            return {
                'source': 'path',
                'requested_family': fallback_family,
                'path': font_path,
                'family': fallback_family,
            }
    return None


def configure_matplotlib_style():
    resolved_font = resolve_cjk_font()
    if resolved_font is not None:
        font_family = resolved_font['family']
        sans_serif_fonts = [font_family] + [
            fallback
            for fallback in DEFAULT_FONT_FALLBACKS
            if fallback != font_family
        ]
        print('Using CJK font: %s' % resolved_font['path'])
    else:
        font_family = 'sans-serif'
        sans_serif_fonts = list(DEFAULT_FONT_FALLBACKS)
        print(
            'Warning: no explicit CJK font was found. '
            'Falling back to default sans-serif fonts; Chinese text may render incorrectly.'
        )

    plt.rcParams.update({
        'font.family': font_family,
        'font.sans-serif': sans_serif_fonts,
        'axes.unicode_minus': False,
        'axes.edgecolor': '#6E7F91',
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def load_subject_display_map(subjects_json_path):
    if not subjects_json_path or not os.path.isfile(subjects_json_path):
        return {}

    with open(subjects_json_path, 'r', encoding='utf-8') as f:
        subjects = json.load(f)

    mapping = {}
    for subject in subjects:
        subject_id = str(subject.get('subject_id', '')).strip()
        if not subject_id:
            continue
        candidate_names = [
            subject.get('name', ''),
            subject.get('subject_name', ''),
        ]
        display_name = ''
        for candidate in candidate_names:
            candidate = str(candidate or '').strip()
            if candidate:
                display_name = normalize_display_name(candidate)
                if display_name:
                    break
        if display_name:
            mapping[subject_id] = display_name
    return mapping


def normalize_display_name(value):
    text = str(value or '').strip()
    if not text:
        return text

    match = re.match(r'^\d{4}_\d{2}_\d{2}_(.+)$', text)
    if match:
        return match.group(1).strip() or text

    parts = text.split('_')
    if len(parts) > 3 and all(part.isdigit() for part in parts[:3]):
        tail = '_'.join(parts[3:]).strip()
        if tail:
            return tail
    return text


def format_subject_label(value):
    display_name = normalize_display_name(value)
    return SUBJECT_LABEL_MAP.get(display_name, display_name)


def subject_sort_key(subject_label):
    if subject_label in SUBJECT_LABEL_ORDER:
        return (0, SUBJECT_LABEL_ORDER.index(subject_label))
    return (1, str(subject_label))


def format_noise_label(noise_type):
    parts = str(noise_type).split('+')
    return '+'.join(NOISE_LABEL_MAP.get(part, part) for part in parts)


def read_metrics_rows(csv_path):
    abs_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError('Required metrics CSV not found: %s' % abs_path)

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing_columns:
            raise ValueError(
                'Metrics CSV missing required column(s) in %s: %s'
                % (abs_path, ', '.join(missing_columns))
            )

        rows = []
        for row in reader:
            normalized = dict(row)
            normalized['subject_id'] = str(normalized.get('subject_id', '')).strip()
            normalized['noise_type'] = str(normalized.get('noise_type', '')).strip()
            metric_value = str(normalized.get('si_sdr_improvement', '')).strip()
            if not normalized['subject_id'] or not normalized['noise_type'] or metric_value == '':
                continue
            try:
                normalized['si_sdr_improvement'] = float(metric_value)
            except ValueError as exc:
                raise ValueError(
                    'Invalid si_sdr_improvement value in %s for subject_id=%s noise_type=%s: %s'
                    % (abs_path, normalized['subject_id'], normalized['noise_type'], metric_value)
                ) from exc
            rows.append(normalized)

    if not rows:
        raise ValueError('Metrics CSV has no valid data rows: %s' % abs_path)
    return rows


def read_optional_metrics_rows(csv_path):
    abs_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        print('Warning: optional metrics CSV not found, skipping: %s' % abs_path)
        return None
    return read_metrics_rows(csv_path)


def ordered_noise_types(noise_types):
    known = [noise for noise in SINGLE_NOISE_ORDER if noise in noise_types]
    extras = sorted(noise for noise in noise_types if noise not in SINGLE_NOISE_ORDER)
    return known + extras


def noise_sort_key(noise_type):
    parts = str(noise_type).split('+')
    indices = []
    for part in parts:
        if part in SINGLE_NOISE_ORDER:
            indices.append(SINGLE_NOISE_ORDER.index(part))
        else:
            indices.append(len(SINGLE_NOISE_ORDER) + 100)
    return (len(parts), indices, str(noise_type))


def group_by_noise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['noise_type']].append(float(row['si_sdr_improvement']))
    return grouped


def build_subject_noise_matrix(rows, subject_display_map, noise_labels):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        raw_display_name = subject_display_map.get(row['subject_id'], row['subject_id'])
        display_name = format_subject_label(raw_display_name)
        grouped[display_name][row['noise_type']].append(float(row['si_sdr_improvement']))

    subjects = sorted(grouped.keys(), key=subject_sort_key)
    matrix = np.full((len(subjects), len(noise_labels)), np.nan, dtype=float)
    for row_idx, subject_name in enumerate(subjects):
        for col_idx, noise_label in enumerate(noise_labels):
            values = grouped[subject_name].get(noise_label, [])
            if values:
                matrix[row_idx, col_idx] = float(np.mean(values))
    return subjects, matrix


def create_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axis(ax, y_grid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#90A4B5')
    ax.spines['bottom'].set_color('#90A4B5')
    if y_grid:
        ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig, output_path, dpi):
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print('Saved figure: %s' % os.path.abspath(output_path))


def plot_bar_chart(rows, output_path, dpi):
    grouped = group_by_noise(rows)
    noise_labels = ordered_noise_types(grouped.keys())
    display_noise_labels = [format_noise_label(noise) for noise in noise_labels]
    mean_values = [float(np.mean(grouped[noise])) for noise in noise_labels]

    fig, ax = create_figure((7.2, 4.5))
    ax.bar(
        display_noise_labels,
        mean_values,
        color=SCIENTIFIC_BLUE,
        edgecolor=SCIENTIFIC_BLUE_DARK,
        linewidth=0.8,
        width=0.65,
    )
    ax.set_xlabel('噪声类型')
    ax.set_ylabel('平均 SI-SDR 提升 / dB')
    style_axis(ax, y_grid=True)
    save_figure(fig, output_path, dpi)


def plot_histogram(rows, output_path, dpi):
    values = [float(row['si_sdr_improvement']) for row in rows]

    fig, ax = create_figure((7.2, 4.5))
    ax.hist(
        values,
        bins='auto',
        color=SCIENTIFIC_BLUE,
        edgecolor=SCIENTIFIC_BLUE_DARK,
        linewidth=0.8,
        alpha=0.9,
    )
    ax.set_xlabel('SI-SDR 提升 / dB')
    ax.set_ylabel('样本数量')
    style_axis(ax, y_grid=True)
    save_figure(fig, output_path, dpi)


def plot_box_chart(rows, output_path, dpi):
    grouped = group_by_noise(rows)
    noise_labels = ordered_noise_types(grouped.keys())
    display_noise_labels = [format_noise_label(noise) for noise in noise_labels]
    values = [grouped[noise] for noise in noise_labels]

    fig, ax = create_figure((7.6, 4.8))
    box = ax.boxplot(
        values,
        labels=display_noise_labels,
        patch_artist=True,
        widths=0.6,
        medianprops={'color': SCIENTIFIC_BLUE_DARK, 'linewidth': 1.2},
        whiskerprops={'color': '#7B90A6', 'linewidth': 1.0},
        capprops={'color': '#7B90A6', 'linewidth': 1.0},
        flierprops={
            'marker': 'o',
            'markerfacecolor': '#BBD0E8',
            'markeredgecolor': SCIENTIFIC_BLUE_DARK,
            'markersize': 3.5,
            'alpha': 0.7,
        },
    )
    for patch in box['boxes']:
        patch.set_facecolor('#BDD0E5')
        patch.set_edgecolor(SCIENTIFIC_BLUE_DARK)
        patch.set_linewidth(1.0)

    ax.set_xlabel('噪声类型')
    ax.set_ylabel('SI-SDR 提升 / dB')
    style_axis(ax, y_grid=True)
    save_figure(fig, output_path, dpi)


def plot_heatmap(rows, output_path, dpi, subject_display_map, x_label):
    noise_labels = sorted({row['noise_type'] for row in rows}, key=noise_sort_key)
    display_noise_labels = [format_noise_label(noise) for noise in noise_labels]
    subjects, matrix = build_subject_noise_matrix(rows, subject_display_map, noise_labels)
    if not subjects:
        raise ValueError('No subject rows available for heatmap: %s' % os.path.abspath(output_path))

    masked = np.ma.masked_invalid(matrix)
    if masked.count() == 0:
        raise ValueError('Heatmap data contains no valid values: %s' % os.path.abspath(output_path))

    fig_width = max(7.2, 0.8 * len(noise_labels) + 3.2)
    fig_height = max(4.8, 0.45 * len(subjects) + 2.0)
    fig, ax = create_figure((fig_width, fig_height))

    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color='white')
    vmin = float(masked.min())
    vmax = float(masked.max())
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    heatmap = ax.imshow(
        masked,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest',
        norm=Normalize(vmin=vmin, vmax=vmax),
    )

    ax.set_xticks(range(len(noise_labels)))
    ax.set_xticklabels(display_noise_labels, rotation=30, ha='right')
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel(x_label)
    ax.set_ylabel('测试人员')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#90A4B5')
    ax.spines['bottom'].set_color('#90A4B5')

    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('平均 SI-SDR 提升 / dB')
    save_figure(fig, output_path, dpi)


def main():
    args = parse_args()
    configure_matplotlib_style()

    metrics_csv = resolve_csv_path(args.metrics_csv, args.eval_dir, BASELINE_METRICS_NAME)
    metrics_12_csv = resolve_csv_path(args.metrics_12_csv, args.eval_dir, DOUBLE_NOISE_METRICS_NAME)
    metrics_13_csv = resolve_csv_path(args.metrics_13_csv, args.eval_dir, TRIPLE_NOISE_METRICS_NAME)

    subject_display_map = load_subject_display_map(args.subjects_json)
    base_rows = read_metrics_rows(metrics_csv)
    ensure_dir(args.output_dir)

    plot_bar_chart(
        base_rows,
        os.path.join(args.output_dir, 'si_sdr_improvement_by_noise_bar.png'),
        args.dpi,
    )
    plot_histogram(
        base_rows,
        os.path.join(args.output_dir, 'si_sdr_improvement_hist.png'),
        args.dpi,
    )
    plot_heatmap(
        base_rows,
        os.path.join(args.output_dir, 'si_sdr_improvement_subject_noise_heatmap.png'),
        args.dpi,
        subject_display_map,
        x_label='噪声类型',
    )
    plot_box_chart(
        base_rows,
        os.path.join(args.output_dir, 'si_sdr_improvement_by_noise_box.png'),
        args.dpi,
    )

    rows_12 = read_optional_metrics_rows(metrics_12_csv)
    if rows_12 is not None:
        plot_heatmap(
            rows_12,
            os.path.join(args.output_dir, 'si_sdr_improvement_subject_noise_combo_heatmap_2noise.png'),
            args.dpi,
            subject_display_map,
            x_label='噪声组合',
        )

    rows_13 = read_optional_metrics_rows(metrics_13_csv)
    if rows_13 is not None:
        plot_heatmap(
            rows_13,
            os.path.join(args.output_dir, 'si_sdr_improvement_subject_noise_combo_heatmap_3noise.png'),
            args.dpi,
            subject_display_map,
            x_label='噪声组合',
        )

    print('Finished generating evaluation figures into %s' % os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()

import argparse
import os
import platform
import sys
from glob import glob


def fail(message):
    print('ERROR: %s' % message)
    return False


def check_directory(path, label):
    if not os.path.isdir(path):
        return fail('%s directory not found: %s' % (label, os.path.abspath(path)))
    print('%s: %s' % (label, os.path.abspath(path)))
    return True


def main():
    parser = argparse.ArgumentParser(description='Check the Platformax JupyterLab environment before GPU training.')
    parser.add_argument('--data-root', default=os.path.join('data', 'raw'), help='Raw subject data directory')
    parser.add_argument('--noise-root', default=os.path.join('data', 'noise'), help='Flat noise wav directory')
    parser.add_argument(
        '--embedder-path',
        default=os.path.join('pretrained', 'embedder.pt'),
        help='Pretrained embedder checkpoint path',
    )
    parser.add_argument('--skip-embedder-check', action='store_true', help='Skip checking the embedder checkpoint')
    parser.add_argument('--device', default='cuda:0', help='CUDA device expected for training')
    parser.add_argument('--require-cuda', action='store_true', help='Fail if CUDA is unavailable')
    args = parser.parse_args()

    ok = True
    print('Python: %s' % sys.version.replace('\n', ' '))
    print('Executable: %s' % sys.executable)
    print('Platform: %s' % platform.platform())
    print('Working directory: %s' % os.getcwd())

    try:
        import torch
    except Exception as exc:
        print('ERROR: failed to import torch: %s' % exc)
        print('The Platformax pytorch image should already provide CUDA-enabled torch.')
        return 1

    print('Torch: %s' % torch.__version__)
    print('CUDA available: %s' % torch.cuda.is_available())
    print('CUDA device count: %d' % torch.cuda.device_count())

    if args.require_cuda and not torch.cuda.is_available():
        ok = fail('CUDA is required but torch.cuda.is_available() is False.') and ok

    if torch.cuda.is_available():
        try:
            device = torch.device(args.device)
            print('Selected device: %s' % device)
            print('Selected GPU: %s' % torch.cuda.get_device_name(device))
        except Exception as exc:
            ok = fail('Cannot use device %s: %s' % (args.device, exc)) and ok

    ok = check_directory(args.data_root, 'Raw data root') and ok
    if os.path.isdir(args.data_root):
        subject_dirs = [
            path for path in glob(os.path.join(args.data_root, '*'))
            if os.path.isdir(path)
        ]
        print('Raw subject directories: %d' % len(subject_dirs))
        if not subject_dirs:
            ok = fail('No subject directories were found under %s.' % os.path.abspath(args.data_root)) and ok

    ok = check_directory(args.noise_root, 'Noise root') and ok
    if os.path.isdir(args.noise_root):
        noise_wavs = glob(os.path.join(args.noise_root, '*.wav'))
        print('Noise wav files: %d' % len(noise_wavs))
        if not noise_wavs:
            ok = fail('No .wav files were found under %s.' % os.path.abspath(args.noise_root)) and ok

    if args.skip_embedder_check:
        print('Embedder checkpoint check: skipped')
    elif os.path.isfile(args.embedder_path):
        size_mb = os.path.getsize(args.embedder_path) / (1024.0 * 1024.0)
        print('Embedder checkpoint: %s (%.2f MB)' % (os.path.abspath(args.embedder_path), size_mb))
    else:
        ok = fail('Embedder checkpoint not found: %s' % os.path.abspath(args.embedder_path)) and ok

    if ok:
        print('Platform environment check passed.')
        return 0
    print('Platform environment check failed.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

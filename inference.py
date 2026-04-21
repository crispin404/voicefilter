import os
import glob
import torch
import argparse

from utils.audio import Audio, load_wav, save_wav
from utils.embedder_checkpoint import DEFAULT_EMBEDDER_PATH, resolve_embedder_path
from utils.hparams import HParam
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def build_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def main(args, hp):
    device = build_device(args.device)
    with torch.no_grad():
        model = VoiceFilter(hp).to(device)
        chkpt_model = torch.load(args.checkpoint_path, map_location=device)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        embedder = SpeechEmbedder(hp).to(device)
        chkpt_embed = torch.load(args.embedder_path, map_location=device)
        embedder.load_state_dict(chkpt_embed)
        embedder.eval()

        audio = Audio(hp)
        dvec_wav, _ = load_wav(args.reference_file, sample_rate=hp.audio.sample_rate, mono=True)
        dvec_mel = audio.get_mel(dvec_wav)
        dvec_mel = torch.from_numpy(dvec_mel).float().to(device)
        dvec = embedder(dvec_mel)
        dvec = dvec.unsqueeze(0)

        mixed_wav, _ = load_wav(args.mixed_file, sample_rate=hp.audio.sample_rate, mono=True)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = torch.from_numpy(mag).float().to(device)

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec)
        est_mag = mag * mask

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'result.wav')
        save_wav(out_path, est_wav, hp.audio.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, default=None,
                        help="path of embedder model pt file, defaults to %s" % DEFAULT_EMBEDDER_PATH)
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_file', type=str, required=True,
                        help='path of reference wav file')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu, cuda, or auto (default)')

    args = parser.parse_args()
    args.embedder_path = resolve_embedder_path(args.embedder_path)

    hp = HParam(args.config)

    main(args, hp)

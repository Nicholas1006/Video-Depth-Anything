"""
KITTI: does a second camera fix the scale of the metric Video-Depth-Anything model?

Uses the KITTI depth-prediction validation split (13 driving sequences) with the repo's own
protocol (benchmark/eval/eval.py, 'kitti'): the first 110 frames of each sequence, images and
ground truth cropped to even sizes, pixels with 0 < depth < 80 m, metrics averaged per frame,
then per sequence. For each sequence it scores:

  original : the metric model's raw output (compare with the README: VDA-S-Metric delta1 = 0.877)
  oracle   : that depth times the single scale that best matches the laser ground truth.
             Uses the ground truth, so it is the best ANY one-number scale fix could do.
  stereo   : that depth times a single scale measured from the left/right cameras
             (Z = f*B/d at matched features). No ground truth used. Needs KITTI's calibration
             files (calib_cam_to_cam.txt per date) for the focal length f and baseline B.

  python experiments/kitti_metric_eval.py --encoder vits --fp32

Data layout (from cvlibs.net/datasets/kitti, free account needed):
  data_depth_annotated/val/<seq>/proj_depth/groundtruth/image_02/*.png    Depth Prediction: data_depth_annotated.zip
  data_depth_seperate/<seq>/<date>/<seq>/image_02|image_03/data/*.png     Raw Data: "synced+rectified" zip per val sequence
  data_depth_seperate/kitti_calib/<date>/calib_cam_to_cam.txt             Raw Data: <date>_calib.zip (one per date)
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402
from utils.stereo_distance import stereo_from_pair  # noqa: E402

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
MAX_DEPTH = 80.0


def find_raw_dir(raw_root, seq):
    date = seq[:10]
    for cand in (os.path.join(raw_root, seq, date, seq), os.path.join(raw_root, date, seq)):
        if os.path.isdir(os.path.join(cand, 'image_02', 'data')):
            return cand
    raise FileNotFoundError(f'raw images for {seq} not found under {raw_root}')


def load_calib(calib_dir, date):
    """Focal length, principal point and stereo baseline of the rectified colour cameras (02 = left, 03 = right)."""
    pats = [os.path.join(calib_dir, date, 'calib_cam_to_cam.txt'),
            os.path.join(calib_dir, '*', date, 'calib_cam_to_cam.txt'),
            os.path.join(calib_dir, '*', '*', date, 'calib_cam_to_cam.txt'),
            os.path.join(calib_dir, f'{date}*', 'calib_cam_to_cam.txt')]
    hits = [h for p in pats for h in glob.glob(p)]
    if not hits:
        return None
    P = {}
    for line in open(hits[0]):
        key, _, vals = line.partition(':')
        if key in ('P_rect_02', 'P_rect_03'):
            P[key] = np.array(vals.split(), dtype=np.float64).reshape(3, 4)
    f = P['P_rect_02'][0, 0]
    return {'focal': f, 'cx': P['P_rect_02'][0, 2], 'cy': P['P_rect_02'][1, 2],
            'baseline': (P['P_rect_02'][0, 3] - P['P_rect_03'][0, 3]) / f, 'file': hits[0]}


def load_sequence(seq, raw_root, depth_root, max_frames):
    gt_dir = os.path.join(depth_root, seq, 'proj_depth', 'groundtruth', 'image_02')
    names = sorted((n for n in os.listdir(gt_dir) if n.endswith('.png')),
                   key=lambda n: int(n.split('.')[0][-4:]))[:max_frames]
    raw = find_raw_dir(raw_root, seq)
    lefts, rights, gts = [], [], []
    for n in names:
        left = cv2.imread(os.path.join(raw, 'image_02', 'data', n))
        right = cv2.imread(os.path.join(raw, 'image_03', 'data', n))
        h, w = left.shape[0] - left.shape[0] % 2, left.shape[1] - left.shape[1] % 2   # repo: even sizes
        gt = cv2.imread(os.path.join(gt_dir, n), -1).astype(np.float32) / 256.0     # repo: factor 256
        lefts.append(left[:h, :w]); rights.append(right[:h, :w]); gts.append(gt[:h, :w][:374, :1242])
    return lefts, rights, np.stack(gts)


def score(pred, gt):
    """AbsRel, RMSE, delta1 as in benchmark/eval/metric.py: per frame, then averaged."""
    valid = (gt > 1e-3) & (gt < MAX_DEPTH)
    pred = np.clip(pred, 1e-3, MAX_DEPTH)
    out = []
    for p, g, m in zip(pred, gt, valid):
        if m.sum() == 0:
            continue
        p, g = p[m].astype(np.float64), g[m].astype(np.float64)
        out.append((np.mean(np.abs(p - g) / g), np.sqrt(np.mean((p - g) ** 2)),
                    np.mean(np.maximum(p / g, g / p) < 1.25)))
    return dict(zip(('abs_rel', 'rmse', 'delta1'), np.mean(out, axis=0).tolist()))


def stereo_scale(lefts, rights, depths, calib, gts, every):
    """One scale factor for the model's depth from matched features in the left/right views (no GT used)."""
    ratios, st_err = [], []
    for i in range(0, len(lefts), every):
        res = stereo_from_pair(lefts[i], rights[i], calib['baseline'], focal_px=calib['focal'],
                               principal_point=(calib['cx'], calib['cy']), dense=False)
        z = res.focal_px * res.baseline_m / res.disparity
        u, v = np.round(res.pts_left).astype(int).T
        ok = (z > 1.0) & (z < MAX_DEPTH) & (u < depths.shape[2]) & (v < depths.shape[1])
        ratios.append(z[ok] / depths[i][v[ok], u[ok]])
        g = gts[i][v[ok], u[ok]]                    # only to REPORT how accurate the stereo points are
        has = g > 1e-3
        st_err.append(np.abs(z[ok][has] / g[has] - 1))
    ratios, st_err = np.concatenate(ratios), np.concatenate(st_err)
    return float(np.median(ratios)), len(ratios), float(np.median(st_err))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_root', default=os.path.join(ROOT, 'data_depth_seperate'))
    p.add_argument('--depth_root', default=os.path.join(ROOT, 'data_depth_annotated', 'val'))
    p.add_argument('--calib_dir', default=None, help='where the unzipped *_calib.zip files are (default: --raw_root)')
    p.add_argument('--encoder', default='vits', choices=list(MODEL_CONFIGS))
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--max_frames', type=int, default=110, help='frames per sequence (repo protocol: 110)')
    p.add_argument('--stereo_every', type=int, default=5, help='use every Nth frame for the stereo scale')
    p.add_argument('--output_dir', default=os.path.join(ROOT, 'outputs', 'kitti_eval'))
    args = p.parse_args()
    calib_dir = args.calib_dir or args.raw_root

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VideoDepthAnything(**MODEL_CONFIGS[args.encoder], metric=True)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints', f'metric_video_depth_anything_{args.encoder}.pth'),
                                     map_location='cpu'), strict=True)
    model = model.to(device).eval()

    seqs = sorted(os.listdir(args.depth_root))
    results = {}
    t0 = time.time()
    for k, seq in enumerate(seqs):
        lefts, rights, gts = load_sequence(seq, args.raw_root, args.depth_root, args.max_frames)
        rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in lefts])
        depths, _ = model.infer_video_depth(rgb, 1, input_size=518, device=device, fp32=args.fp32)
        depths = np.asarray(depths, dtype=np.float32)[:, :gts.shape[1], :gts.shape[2]]

        valid = (gts > 1e-3) & (gts < MAX_DEPTH)
        oracle = float(np.median(gts[valid] / depths[valid]))
        r = {'frames': len(lefts), 'original': score(depths, gts),
             'oracle_scale': oracle, 'oracle': score(depths * oracle, gts)}
        calib = load_calib(calib_dir, seq[:10])
        if calib is not None:
            s, n, st_err = stereo_scale(lefts, rights, depths, calib, gts, args.stereo_every)
            r.update({'calib': calib, 'stereo_scale': s, 'stereo_points': n,
                      'stereo_point_median_err': st_err, 'stereo': score(depths * s, gts)})
        results[seq] = r
        line = (f'[{k + 1:2d}/{len(seqs)}] {seq}  {len(lefts):3d} frames  delta1 original {r["original"]["delta1"]:.3f}'
                f'  oracle {r["oracle"]["delta1"]:.3f} (x{oracle:.2f})')
        if 'stereo' in r:
            line += f'  stereo {r["stereo"]["delta1"]:.3f} (x{r["stereo_scale"]:.2f})'
        print(line + f'  [{time.time() - t0:.0f}s]', flush=True)

    variants = [v for v in ('original', 'oracle', 'stereo') if all(v in r for r in results.values())]
    print(f'\nKITTI val, {len(results)} sequences, metric {args.encoder} {"fp32" if args.fp32 else "fp16"} '
          f'(mean over sequences, as benchmark/eval/eval.py)')
    print(f'{"":10s} {"AbsRel":>8s} {"RMSE (m)":>9s} {"delta1":>8s}')
    summary = {}
    for v in variants:
        m = {key: float(np.mean([r[v][key] for r in results.values()])) for key in ('abs_rel', 'rmse', 'delta1')}
        summary[v] = m
        print(f'{v:10s} {m["abs_rel"]:8.3f} {m["rmse"]:9.2f} {m["delta1"]:8.3f}')
    if 'stereo' not in variants:
        print('\n(stereo skipped: no calib_cam_to_cam.txt found - unzip the KITTI *_calib.zip files under --calib_dir)')
    else:
        print(f'\nstereo points vs laser ground truth: median error '
              f'{100 * np.median([r["stereo_point_median_err"] for r in results.values()]):.1f}%')

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f'kitti_metric_{args.encoder}_{"fp32" if args.fp32 else "fp16"}.json')
    with open(out, 'w') as fh:
        json.dump({'args': vars(args), 'summary': summary, 'sequences': results}, fh, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()

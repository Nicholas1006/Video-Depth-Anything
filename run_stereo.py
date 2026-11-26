"""
Measure things using two shots from a side-by-side camera rig with a known baseline.

  # size and position of a person (baseline only, no focal length needed)
  python run_stereo.py --left a.jpg --right b.jpg --baseline 0.30 --box 500 200 640 650

  # one-off: something measured with a tape to be 3.0 m away gives the rig's focal length...
  python run_stereo.py --left a.jpg --right b.jpg --baseline 0.30 --box 600 300 700 400 --calibrate 3.0
  # ...which then also gives distances for every later shot from the same rig
  python run_stereo.py --left camA.mp4 --right camB.mp4 --frame 120 --baseline 0.30 --focal 1012 --select

  # a side-by-side 3D video (both views in one frame), one view shifted for viewing comfort
  python run_stereo.py --left "SBS video.mp4" --sbs --auto_offset --frame 570 --baseline 0.065 --select

Coordinates are pixels in the left image (the left half for --sbs).
"""
import argparse
import os

import cv2
import numpy as np

from utils.stereo_distance import (VIDEO_EXTS, colorize_disparity, disparity_range, draw_measurements,
                                   estimate_disparity_offset, focal_from_hfov, focal_from_known_distance,
                                   load_frame, split_sbs, stereo_from_pair)


def select_boxes(image, max_display=1280):
    s = min(1.0, max_display / max(image.shape[:2]))
    disp = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
    rois = cv2.selectROIs('Drag a box around each object, ENTER after each, ESC when done', disp, False)
    cv2.destroyAllWindows()
    return [(x / s, y / s, (x + w) / s, (y + h) / s) for x, y, w, h in np.asarray(rois).reshape(-1, 4) if w and h]


class AddQuery(argparse.Action):
    """--point / --box append to one list so their command-line order is kept (--calibrate uses the first)."""
    def __call__(self, parser, namespace, values, option_string=None):
        queries = list(getattr(namespace, self.dest) or [])
        queries.append((self.const, tuple(values)))
        setattr(namespace, self.dest, queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stereo measurements with a known baseline')
    parser.add_argument('--left', required=True, help='left image or video')
    parser.add_argument('--right', help='right image or video (not needed with --sbs)')
    parser.add_argument('--sbs', action='store_true', help='--left is a side-by-side 3D image/video holding both views')
    parser.add_argument('--half_sbs', action='store_true', help='like --sbs, but each view is squeezed to half width')
    offset = parser.add_mutually_exclusive_group()
    offset.add_argument('--disparity_offset', type=float, default=0.0,
                        help='pixels to add to every shift, if one view of a 3D video was shifted sideways')
    offset.add_argument('--auto_offset', action='store_true',
                        help='estimate that shift, assuming the farthest things in view are effectively at infinity')
    parser.add_argument('--frame', type=int, default=0, help='frame index when the inputs are videos')
    parser.add_argument('--right_frame', type=int, default=None, help='frame index in the right video if different')
    parser.add_argument('--baseline', type=float, required=True, help='distance between the two cameras, metres')
    focal = parser.add_mutually_exclusive_group()
    focal.add_argument('--focal', type=float, help='focal length in pixels at the input resolution (for distances)')
    focal.add_argument('--hfov', type=float, help='horizontal field of view in degrees (for distances)')
    focal.add_argument('--calibrate', type=float, metavar='DIST',
                       help='the first --point/--box is DIST metres from the left camera (straight line); '
                            'work out the focal length from it')
    parser.add_argument('--point', type=float, nargs=2, action=AddQuery, const='point', dest='queries',
                        metavar=('X', 'Y'), help='measure around this pixel (repeatable)')
    parser.add_argument('--box', type=float, nargs=4, action=AddQuery, const='box', dest='queries',
                        metavar=('X0', 'Y0', 'X1', 'Y1'), help='measure this box (repeatable)')
    parser.add_argument('--select', action='store_true', help='draw boxes with the mouse')
    parser.add_argument('--no_dense', action='store_true', help='skip the SGBM dense disparity map')
    parser.add_argument('--max_res', type=int, default=1600)
    parser.add_argument('--output_dir', type=str, default='./outputs/stereo')
    parser.add_argument('--save_npz', action='store_true', help='save matches, shifts and the dense map')
    args = parser.parse_args()

    sbs = args.sbs or args.half_sbs
    is_video = os.path.splitext(args.left)[1].lower() in VIDEO_EXTS
    if not sbs and not args.right:
        parser.error('--right is required unless --sbs / --half_sbs is given')

    def get_pair(frame_index):
        if sbs:
            return split_sbs(load_frame(args.left, frame_index), half=args.half_sbs)
        rf = frame_index if args.right_frame is None else args.right_frame + (frame_index - args.frame)
        return load_frame(args.left, frame_index), load_frame(args.right, rf)

    left, right = get_pair(args.frame)
    h, w = left.shape[:2]
    focal_px = args.focal if args.focal is not None else (focal_from_hfov(w, args.hfov) if args.hfov else None)

    disparity_offset = args.disparity_offset
    if args.auto_offset:
        if is_video:  # sample frames within +-5 s of the requested one
            cap = cv2.VideoCapture(args.left)
            fps, n = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            idx = np.unique(np.linspace(max(0, args.frame - 5 * fps), min(n - 1, args.frame + 5 * fps), 10).astype(int))
            pairs = [get_pair(int(i)) for i in idx]
        else:
            pairs = [(left, right)]
        disparity_offset = estimate_disparity_offset(pairs)
        print(f'estimated disparity offset: {disparity_offset:+.2f} px (from the farthest matched points)')

    res = stereo_from_pair(left, right, args.baseline, focal_px, dense=not args.no_dense, max_res=args.max_res,
                           disparity_offset_px=disparity_offset)
    lo, hi = disparity_range(res)
    print(f'image {w}x{h}, baseline {args.baseline} m' + (f', focal {focal_px:.1f} px' if focal_px else ''))
    print(f'{len(res.pts_left)} of {res.n_matches} feature matches lie on the same row '
          f'(median vertical offset {res.row_offset_px:+.2f} px)')
    print(f'shifts between the shots: {lo:.1f} - {hi:.1f} px (2nd-98th percentile), i.e. '
          f'{1000 * args.baseline / hi:.2f} - {1000 * args.baseline / lo:.2f} mm per pixel')
    for msg in res.warnings:
        print('WARNING:', msg)

    queries = list(args.queries or [])
    if args.select:
        queries += [('box', b) for b in select_boxes(left)]
    measurements = [res.measure(**{kind: q}) for kind, q in queries]

    if args.calibrate is not None:
        if not measurements:
            parser.error('--calibrate needs a --point or --box on the object at the known distance')
        ref = measurements[0]
        ahead_sq = args.calibrate ** 2 - ref.offset_m[0] ** 2 - ref.offset_m[1] ** 2
        if ahead_sq <= 0:
            parser.error(f'the reference is already {np.hypot(*ref.offset_m):.2f} m off-axis, '
                         f'so it cannot be {args.calibrate} m away')
        res.focal_px = focal_from_known_distance(ref.disparity_px, np.sqrt(ahead_sq), args.baseline)
        print(f'\nCALIBRATED focal length: {res.focal_px:.1f} px at {w}x{h} '
              f'(+-{100 * ref.rel_uncertainty:.1f}%). Pass --focal {res.focal_px:.1f} for this rig from now on.')
        measurements = [res.measure(**{kind: q}) for kind, q in queries]

    print()
    for m in measurements:
        print(m)

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.left))[0]
    if is_video:
        name += f'_f{args.frame}'
    vis = draw_measurements(left, res, measurements)
    if res.disparity_map is not None:
        vis = np.vstack([vis, colorize_disparity(res.disparity_map, lo, hi)])
    out_png = os.path.join(args.output_dir, f'{name}_stereo.png')
    cv2.imwrite(out_png, vis)
    print('\nsaved', out_png)
    if args.save_npz:
        out_npz = os.path.join(args.output_dir, f'{name}_stereo.npz')
        np.savez_compressed(out_npz, pts_left=res.pts_left, disparity=res.disparity,
                            disparity_map=res.disparity_map if res.disparity_map is not None else np.empty(0),
                            baseline=args.baseline, principal_point=res.principal_point,
                            focal=res.focal_px if res.focal_px else np.nan)
        print('saved', out_npz)

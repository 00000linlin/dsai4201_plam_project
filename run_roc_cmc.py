import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def gabor_array(roi_size: int, sigma: float, wavelength: float, ratio: float) -> np.ndarray:
    half_length = 17
    xmin, xmax = -half_length, half_length
    ymin, ymax = -half_length, half_length
    x, y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))

    out_rows = roi_size + 34
    out_cols = roi_size + 34

    yy, xx = np.meshgrid(np.arange(35), np.arange(35), indexing="ij")
    mask = ((yy - 17) ** 2 + (xx - 17) ** 2 <= 289).astype(np.float64)

    gabor = np.zeros((out_rows, out_cols, 6), dtype=np.complex128)
    for ori_index in range(6):
        theta = np.pi / 6.0 * ori_index
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gb = np.exp(-0.5 * (x_theta**2 / sigma**2 + y_theta**2 / (ratio * sigma) ** 2))
        gb = gb * np.cos(2 * np.pi / wavelength * x_theta)

        mean_inner = np.sum(gb * mask) / np.sum(mask)
        gb = (gb - mean_inner) * mask

        gabor[:, :, ori_index] = np.fft.fft2(gb, s=(out_rows, out_cols))

    return gabor


def compete_code(image: np.ndarray, gabor: np.ndarray) -> np.ndarray:
    image_rows, image_cols = image.shape
    out_rows = image_rows + 34
    out_cols = image_cols + 34

    image_fft = np.fft.fft2(image, s=(out_rows, out_cols))
    responses = np.zeros((image_rows, image_cols, 6), dtype=np.float64)

    rs = (out_rows - image_rows) // 2
    cs = (out_cols - image_cols) // 2

    for ori_index in range(6):
        conv_res = np.fft.ifft2(image_fft * gabor[:, :, ori_index])
        raw = conv_res[rs : rs + image_rows, cs : cs + image_cols]
        responses[:, :, ori_index] = np.real(raw)

    # MATLAB index is 1..6
    return np.argmin(responses, axis=2).astype(np.int32) + 1


def create_cc_feature(image: np.ndarray, gabor: np.ndarray, patch_size: int) -> np.ndarray:
    cc_map = compete_code(image, gabor)
    rows = cc_map.shape[0]
    patches_per_row = rows // patch_size

    cut = (rows - patches_per_row * patch_size) // 2
    cc_map = cc_map[cut : rows - cut, cut : rows - cut]

    feat = []
    for patch_row in range(patches_per_row):
        for patch_col in range(patches_per_row):
            r0 = patch_row * patch_size
            r1 = (patch_row + 1) * patch_size
            c0 = patch_col * patch_size
            c1 = (patch_col + 1) * patch_size
            patch = cc_map[r0:r1, c0:c1].ravel()
            hist = np.bincount(patch, minlength=7)[1:7].astype(np.float64)
            feat.append(hist / (patch_size**2))

    return np.concatenate(feat, axis=0)


def load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def compute_roc(genuine_scores: np.ndarray, impostor_scores: np.ndarray, num_thresholds: int = 2000):
    score_min = min(np.min(genuine_scores), np.min(impostor_scores))
    score_max = max(np.max(genuine_scores), np.max(impostor_scores))
    thresholds = np.linspace(score_min, score_max, num_thresholds)

    far = np.zeros(num_thresholds, dtype=np.float64)
    gar = np.zeros(num_thresholds, dtype=np.float64)

    for i, t in enumerate(thresholds):
        far[i] = np.mean(impostor_scores >= t)
        gar[i] = np.mean(genuine_scores >= t)

    return far, gar, thresholds


def evaluate(
    gallery_dir: Path,
    probe_dir: Path,
    out_dir: Path,
    patch_size: int,
    lambda_: float,
    sigma: float,
    ratio: float,
    wavelength: float,
    samples_per_palm_per_session: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery_files = sorted(gallery_dir.glob("*.bmp"), key=lambda p: p.name)
    probe_files = sorted(probe_dir.glob("*.bmp"), key=lambda p: p.name)

    if len(gallery_files) == 0 or len(probe_files) == 0:
        raise RuntimeError("No .bmp files found in gallery/probe folders.")

    if len(gallery_files) != len(probe_files):
        raise RuntimeError("Gallery and probe sizes do not match.")

    num_gallery = len(gallery_files)
    num_probe = len(probe_files)

    if num_gallery % samples_per_palm_per_session != 0:
        raise RuntimeError("Gallery size is not divisible by samples_per_palm_per_session.")

    num_classes = num_gallery // samples_per_palm_per_session

    sample_image = load_gray_image(gallery_files[0])
    if sample_image.shape[0] != sample_image.shape[1]:
        raise RuntimeError("ROI image must be square.")
    roi_size = sample_image.shape[0]

    print(f"Gallery dir: {gallery_dir}")
    print(f"Probe dir:   {probe_dir}")
    print(f"Gallery images: {num_gallery}")
    print(f"Probe images:   {num_probe}")
    print(f"Classes (palms): {num_classes}")
    print(f"ROI size: {roi_size}x{roi_size}")

    gabor = gabor_array(roi_size, sigma, wavelength, ratio)
    feat_dim = (roi_size // patch_size) ** 2 * 6

    print("Building gallery dictionary...")
    dic = np.zeros((feat_dim, num_gallery), dtype=np.float64)
    for i, p in enumerate(gallery_files):
        im = load_gray_image(p)
        fv = create_cc_feature(im, gabor, patch_size)
        n = np.linalg.norm(fv)
        if n > 0:
            fv = fv / n
        dic[:, i] = fv
        if (i + 1) % 500 == 0:
            print(f"  gallery processed: {i + 1}/{num_gallery}")

    p_mat = np.linalg.solve(dic.T @ dic + lambda_ * np.eye(num_gallery), dic.T)

    print("Scoring probes...")
    all_class_scores = np.zeros((num_probe, num_classes), dtype=np.float64)
    probe_ranks = np.zeros(num_probe, dtype=np.int32)
    genuine_scores = np.zeros(num_probe, dtype=np.float64)

    for i, p in enumerate(probe_files):
        im = load_gray_image(p)
        y = create_cc_feature(im, gabor, patch_size)
        ny = np.linalg.norm(y)
        if ny > 0:
            y = y / ny

        x0 = p_mat @ y

        residual = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            start = c * samples_per_palm_per_session
            end = (c + 1) * samples_per_palm_per_session
            partial_dic = dic[:, start:end]
            partial_x0 = x0[start:end]
            d = partial_dic @ partial_x0 - y
            residual[c] = np.sum(d**2)

        scores = -residual
        all_class_scores[i, :] = scores

        gt_class = i // samples_per_palm_per_session
        genuine_scores[i] = scores[gt_class]

        order = np.argsort(-scores)
        probe_ranks[i] = int(np.where(order == gt_class)[0][0]) + 1

        if (i + 1) % 250 == 0:
            print(f"  probe processed: {i + 1}/{num_probe}")

    max_rank = num_classes
    cmc = np.array([np.mean(probe_ranks <= r) for r in range(1, max_rank + 1)], dtype=np.float64)

    impostor_mask = np.ones_like(all_class_scores, dtype=bool)
    for i in range(num_probe):
        gt_class = i // samples_per_palm_per_session
        impostor_mask[i, gt_class] = False
    impostor_scores = all_class_scores[impostor_mask]

    far, gar, thresholds = compute_roc(genuine_scores, impostor_scores, num_thresholds=2000)

    frr = 1.0 - gar
    eer_idx = np.argmin(np.abs(far - frr))
    eer = 0.5 * (far[eer_idx] + frr[eer_idx])

    rank1 = float(cmc[0])
    rank5 = float(cmc[min(4, max_rank - 1)])
    rank10 = float(cmc[min(9, max_rank - 1)])

    print(f"Rank-1:  {rank1:.4f}")
    print(f"Rank-5:  {rank5:.4f}")
    print(f"Rank-10: {rank10:.4f}")
    print(f"EER:     {eer:.4f}")

    plt.figure(figsize=(8, 5.6), dpi=180)
    ax = plt.gca()
    ax.plot(far, gar, color="#0B5ED7", linewidth=2.4, label="ROC")
    ax.fill_between(far, gar, 0.9, color="#0B5ED7", alpha=0.08)
    ax.scatter([far[eer_idx]], [gar[eer_idx]], color="#D62828", s=34, zorder=3,
               label=f"EER={eer:.4f}")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("FAR")
    ax.set_ylabel("GAR")
    ax.set_title("ROC Curve (CRC-CompCode)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.9, 1.001)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # FAR is heavily concentrated near 0 for strong systems.
    # Add log-scale and zoomed ROC plots for better visual discrimination.
    far_for_log = np.clip(far, 1e-6, 1.0)

    plt.figure(figsize=(8, 5.6), dpi=180)
    ax = plt.gca()
    ax.semilogx(far_for_log, gar, color="#146C43", linewidth=2.4)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("FAR (log scale)")
    ax.set_ylabel("GAR")
    ax.set_title("ROC Curve (Log-X, CRC-CompCode)")
    ax.set_xlim(1e-6, 1.0)
    ax.set_ylim(0.9, 1.001)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    roc_logx_path = out_dir / "roc_curve_logx.png"
    plt.tight_layout()
    plt.savefig(roc_logx_path)
    plt.close()

    plt.figure(figsize=(8, 5.6), dpi=180)
    ax = plt.gca()
    ax.plot(far, gar, color="#8B5E34", linewidth=2.4)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("FAR")
    ax.set_ylabel("GAR")
    ax.set_title("ROC Curve (Zoomed, CRC-CompCode)")
    ax.set_xlim(0.0, 0.02)
    ax.set_ylim(0.9, 1.001)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    roc_zoom_path = out_dir / "roc_curve_zoom.png"
    plt.tight_layout()
    plt.savefig(roc_zoom_path)
    plt.close()

    plt.figure(figsize=(7, 5), dpi=140)
    plt.plot(np.arange(1, max_rank + 1), cmc, linewidth=2)
    plt.grid(True)
    plt.xlabel("Rank")
    plt.ylabel("Recognition Rate")
    plt.title("CMC Curve (CRC-CompCode)")
    plt.xlim(1, min(100, max_rank))
    cmc_path = out_dir / "cmc_curve.png"
    plt.tight_layout()
    plt.savefig(cmc_path)
    plt.close()

    np.savez(
        out_dir / "roc_cmc_results.npz",
        far=far,
        gar=gar,
        thresholds=thresholds,
        cmc=cmc,
        probe_ranks=probe_ranks,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        eer=eer,
        rank1=rank1,
        rank5=rank5,
        rank10=rank10,
    )

    print(f"Saved: {roc_path}")
    print(f"Saved: {roc_logx_path}")
    print(f"Saved: {roc_zoom_path}")
    print(f"Saved: {cmc_path}")
    print(f"Saved: {out_dir / 'roc_cmc_results.npz'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ROC/CMC evaluation for contactless palmprint ROI data.")
    default_gallery = Path(__file__).resolve().parent.parent / "roi" / "session1"
    default_probe = Path(__file__).resolve().parent.parent / "roi" / "session2"
    default_out = Path(__file__).resolve().parent / "results_py"

    parser.add_argument("--gallery", type=Path, default=default_gallery)
    parser.add_argument("--probe", type=Path, default=default_probe)
    parser.add_argument("--out", type=Path, default=default_out)

    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.35)
    parser.add_argument("--sigma", type=float, default=4.85)
    parser.add_argument("--ratio", type=float, default=1.92)
    parser.add_argument("--wavelength", type=float, default=14.1)
    parser.add_argument("--samples-per-palm", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        gallery_dir=args.gallery,
        probe_dir=args.probe,
        out_dir=args.out,
        patch_size=args.patch_size,
        lambda_=args.lambda_,
        sigma=args.sigma,
        ratio=args.ratio,
        wavelength=args.wavelength,
        samples_per_palm_per_session=args.samples_per_palm,
    )

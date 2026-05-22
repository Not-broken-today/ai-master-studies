import os, sys, csv, time, hashlib, torch, numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

DARKIR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DarkIR')
if DARKIR_ROOT not in sys.path:
    sys.path.insert(0, DARKIR_ROOT)

from DarkIR.archs.DarkIR import DarkIR

INPUT_DIR   = 'assets/inputs'
AUTHOR_DIR  = 'assets/results'

models_to_test = [
    ('models/DarkIR_1k_cr_mt.pt', 'Full_1k')
]

def calc_metrics(img1, img2):
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse < 1e-10:
        p_val = ">= 99.99"
        p_num = 99.99
    else:
        p_val = str(round(10 * np.log10((255**2) / mse), 2))
        p_num = float(p_val)
    s_val = str(round(ssim(arr1, arr2, data_range=255, channel_axis=-1), 4))
    return p_val, s_val, p_num, float(s_val)

def get_md5(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    print("=" * 75)
    print("    DarkIR: Multi-Model Quantitative Evaluation Pipeline")
    print("=" * 75)
    
    device = torch.device('cpu')
    print(f"\n[INFO] Target device: {device}")

    img_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not img_files:
        print("[ERROR] No input images found.")
        return
    print(f"[INFO] Found {len(img_files)} images.\n")

    all_reports = []
    summary_data = []

    for ckpt_path, model_tag in models_to_test:
        print("-" * 75)
        print(f"[INFO] Evaluating: {model_tag} ({os.path.basename(ckpt_path)})")
        
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint not found: {ckpt_path}. Skipping.\n")
            continue

        model = DarkIR()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('params', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state)
        model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Loaded. Trainable parameters: {n_params:,}\n")

        out_dir = f'assets/results_{model_tag}'
        os.makedirs(out_dir, exist_ok=True)

        model_report = []
        total_time = 0.0
        ref_psnr_nums, ref_ssim_nums = [], []
        enh_psnr_nums, enh_ssim_nums = [], []

        print(f"Processing images ({model_tag}):")
        for i, fname in enumerate(img_files, 1):
            img_path = os.path.join(INPUT_DIR, fname)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            t0 = time.time()
            with torch.no_grad():
                out = model(tensor)
                if isinstance(out, (list, tuple)): out = out[0]
            elapsed = time.time() - t0
            total_time += elapsed

            out = torch.clamp(out, 0., 1.)
            res_img = transforms.ToPILImage()(out.squeeze(0))
            save_path = os.path.join(out_dir, fname)
            res_img.save(save_path)

            auth_path = os.path.join(AUTHOR_DIR, fname)
            ref_p, ref_s, ref_p_num, ref_s_num = "N/A", "N/A", None, None
            md5 = "N/A"
            if os.path.exists(auth_path):
                auth_img = Image.open(auth_path).convert('RGB')
                ref_p, ref_s, ref_p_num, ref_s_num = calc_metrics(auth_img, res_img)
                md5 = str(get_md5(auth_path) == get_md5(save_path))

            enh_p, enh_s, enh_p_num, enh_s_num = calc_metrics(img, res_img)

            print(f"  [{i}/{len(img_files)}] {fname:<10} | {w}x{h:<4} | Time: {elapsed:.2f}s")
            print(f"    Ref vs Out -> PSNR: {ref_p:<8} dB | SSIM: {ref_s:<6} | MD5: {md5}")
            print(f"    In  vs Out -> PSNR: {enh_p:<8} dB | SSIM: {enh_s:<6}")

            if ref_p_num is not None:
                ref_psnr_nums.append(ref_p_num)
                ref_ssim_nums.append(ref_s_num)
            enh_psnr_nums.append(enh_p_num)
            enh_ssim_nums.append(enh_s_num)

            model_report.append([model_tag, fname, ref_p, ref_s, md5, enh_p, enh_s, f"{elapsed:.3f}"])

        all_reports.extend(model_report)
        
        avg_time = total_time / len(img_files) if img_files else 0
        avg_rp = np.mean(ref_psnr_nums) if ref_psnr_nums else 0
        avg_rs = np.mean(ref_ssim_nums) if ref_ssim_nums else 0
        avg_ep = np.mean(enh_psnr_nums) if enh_psnr_nums else 0
        avg_es = np.mean(enh_ssim_nums) if enh_ssim_nums else 0
        
        summary_data.append({
            'tag': model_tag, 'params': n_params,
            'avg_rp': avg_rp, 'avg_rs': avg_rs,
            'avg_ep': avg_ep, 'avg_es': avg_es,
            'avg_t': avg_time
        })
        print(f"\n[INFO] {model_tag} finished. Avg inference: {avg_time:.2f}s/image\n")

    csv_path = 'evaluation_report.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Image', 'Ref_PSNR_dB', 'Ref_SSIM', 'MD5_Match', 'Enh_PSNR_dB', 'Enh_SSIM', 'Time_s'])
        writer.writerows(all_reports)

    print("=" * 75)
    print("AGGREGATED RESULTS")
    print("=" * 75)
    print(f"{'Model':<12} | {'Params':>10} | {'Avg Ref PSNR':>12} | {'Avg Ref SSIM':>12} | {'Avg Enh PSNR':>12} | {'Avg Time (s)':>12}")
    print("-" * 75)
    for d in summary_data:
        rp_str = f">= 99.99 dB" if d['avg_rp'] >= 99 else f"{d['avg_rp']:.2f} dB"
        print(f"{d['tag']:<12} | {d['params']:>10,} | {rp_str:>12} | {d['avg_rs']:>12.4f} | {d['avg_ep']:>12.2f} dB | {d['avg_t']:>12.3f}")
    print("=" * 75)
    print(f"[INFO] Full report exported to: {csv_path}")

if __name__ == '__main__':
    main()
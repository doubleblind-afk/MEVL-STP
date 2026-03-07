"""
Download CTW1500 trained models from Modal volumes.

Downloads:
  - Detection checkpoint: v28_sota/20260301_202746/epoch_30.pth
  - VLM LoRA adapter: qwen_vlm_stageA_gt_v8/20260210_141051/final_adapter/
"""
import modal
import os
from pathlib import Path

app = modal.App("download-ctw1500-models")
output_volume = modal.Volume.from_name("vlm-outputs", create_if_missing=False)

@app.function(
    volumes={"/output": output_volume},
    timeout=3600,
)
def download_models():
    import shutil

    # Create local output directory
    local_dir = Path("/tmp/ctw1500_models")
    local_dir.mkdir(exist_ok=True)

    # 1. Download detection checkpoint
    det_src = "/output/v28_sota/20260301_202746/epoch_30.pth"
    det_dst = local_dir / "ctw1500_detection_epoch30.pth"

    if os.path.exists(det_src):
        print(f"Copying detection checkpoint: {det_src}")
        shutil.copy(det_src, det_dst)
        print(f"  Saved to: {det_dst}")
        print(f"  Size: {os.path.getsize(det_dst) / (1024**2):.1f} MB")
    else:
        print(f"ERROR: Detection checkpoint not found: {det_src}")
        return None

    # 2. Download VLM LoRA adapter (all files in directory)
    adapter_src = "/output/qwen_vlm_stageA_gt_v8/20260210_141051/final_adapter"
    adapter_dst = local_dir / "ctw1500_vlm_lora_adapter"

    if os.path.exists(adapter_src):
        print(f"\nCopying LoRA adapter: {adapter_src}")
        shutil.copytree(adapter_src, adapter_dst, dirs_exist_ok=True)

        # List files
        files = list(adapter_dst.rglob("*"))
        print(f"  Copied {len([f for f in files if f.is_file()])} files:")
        for f in sorted(files):
            if f.is_file():
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"    {f.name}: {size_mb:.1f} MB")
    else:
        print(f"ERROR: LoRA adapter not found: {adapter_src}")
        # Try best_adapter instead
        adapter_src_alt = "/output/qwen_vlm_stageA_gt_v8/20260210_141051/best_adapter"
        if os.path.exists(adapter_src_alt):
            print(f"  Found best_adapter instead, using: {adapter_src_alt}")
            shutil.copytree(adapter_src_alt, adapter_dst, dirs_exist_ok=True)
        else:
            return None

    print(f"\nAll files ready in: {local_dir}")
    print("\nTo download to your local machine, the Modal CLI will save them to:")
    print("  ./ctw1500_detection_epoch30.pth")
    print("  ./ctw1500_vlm_lora_adapter/")

    return str(local_dir)


@app.local_entrypoint()
def main():
    result = download_models.remote()
    if result:
        print(f"\n✅ Download complete: {result}")
        print("\nFiles are ready in Modal's /tmp. Use modal volume get to download them locally.")
    else:
        print("\n❌ Download failed. Check the paths above.")

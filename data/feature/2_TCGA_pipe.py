import os
import csv
import math
import shutil
import subprocess
import glob
from multiprocessing import Process, Queue

CONFIG = {
    # TODO: Set paths before running
    "original_manifest": "./data/TCGA/Liver/Liver.txt",
    "gdc_client_bin": "./tools/gdc-client",
    "pre_feature_script": "./data/feature/pre_feature.py",
    "augment_feature_script": "./data/feature/augment_feature.py",
    "base_tmp_dir": "./tmp/TCGA_processing",
    "output_dir": "./output/features",
    "use_augment": True,
    "model_path": "./models/CONCH/pytorch_model.bin",
    "feature_batch_size": 64,
    "patch_size": 512,
    "save_vis": False,
    "num_workers": 15,
    "cuda_devices": [0, 1, 2],
}


def run_command(cmd, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, shell=True, check=True, env=full_env)


def clean_directory(target_dir):
    if not os.path.exists(target_dir):
        return
    for item in os.listdir(target_dir):
        path = os.path.join(target_dir, item)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")


def get_existing_stems(output_dir, use_augment=False):
    """
    Get the set of already processed file names
    
    Args:
        output_dir: Output directory
        use_augment: Whether augmentation is used
            - False: Check if {stem}.pt exists
            - True: Check if all 9 augmentation versions exist
    """
    if not os.path.exists(output_dir):
        return set()
    
    if not use_augment:
        return {os.path.splitext(f)[0] for f in os.listdir(output_dir) if not f.startswith('.')}
    
    augment_suffixes = [
        '_original',
        '_tl', '_tr', '_bl', '_br',
        '_tl_flip', '_tr_flip', '_bl_flip', '_br_flip'
    ]
    
    all_files = {f for f in os.listdir(output_dir) if f.endswith('.pt') and not f.startswith('.')}
    
    base_names = set()
    for suffix in augment_suffixes:
        for f in all_files:
            if f.endswith(f"{suffix}.pt"):
                base_name = f[:-len(f"{suffix}.pt")]
                base_names.add(base_name)
    
    complete_stems = set()
    for base_name in base_names:
        all_exist = all(
            f"{base_name}{suffix}.pt" in all_files
            for suffix in augment_suffixes
        )
        if all_exist:
            complete_stems.add(base_name)
    
    return complete_stems


def worker(worker_id, task_queue, config, cuda_device):
    work_dir = os.path.join(config["base_tmp_dir"], f"worker_{worker_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    env = {"CUDA_VISIBLE_DEVICES": str(cuda_device)}
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        entry, fieldnames = task
        svs_filename = entry.get('filename', '')
        print(f"[Worker {worker_id}] Processing: {svs_filename}")
        
        clean_directory(work_dir)
        
        manifest_path = os.path.join(work_dir, "manifest.txt")
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(entry)
        
        try:
            run_command(f"{config['gdc_client_bin']} download -m {manifest_path} -d {work_dir}", env)
            
            for svs_path in glob.glob(os.path.join(work_dir, "*", "*.svs")):
                shutil.move(svs_path, os.path.join(work_dir, os.path.basename(svs_path)))
            
            if config.get("use_augment", False):
                cmd = (
                    f"python {config['augment_feature_script']} "
                    f"--input {work_dir} --output {config['output_dir']} "
                    f"--model {config['model_path']} "
                    f"--batch-size {config['feature_batch_size']} "
                    f"--patch-size {config['patch_size']}"
                )
                if config.get("save_vis", False):
                    vis_dir = os.path.join(config['output_dir'], "thumbnails")
                    cmd += f" --save-vis --vis-dir {vis_dir}"
            else:
                cmd = (
                    f"python {config['pre_feature_script']} "
                    f"--input {work_dir} --output {config['output_dir']} "
                    f"--model {config['model_path']} "
                    f"--batch-size {config['feature_batch_size']} "
                    f"--patch-size {config['patch_size']}"
                )
            
            run_command(cmd, env)
            print(f"[Worker {worker_id}] Done: {svs_filename}")
            
        except Exception as e:
            print(f"[Worker {worker_id}] Error processing {svs_filename}: {e}")
        
        finally:
            clean_directory(work_dir)
    
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[Worker {worker_id}] Exiting")


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["base_tmp_dir"], exist_ok=True)
    
    mode = "Augmented Feature Extraction" if CONFIG.get("use_augment", False) else "Normal Feature Extraction"
    print(f"Feature extraction mode: {mode}")
    if CONFIG.get("use_augment", False) and CONFIG.get("save_vis", False):
        print(f"Thumbnail saving: Enabled")
    print()
    
    with open(CONFIG["original_manifest"], 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        entries = list(reader)
    
    use_augment = CONFIG.get("use_augment", False)
    existing = get_existing_stems(CONFIG["output_dir"], use_augment=use_augment)
    pending = [e for e in entries if os.path.splitext(e.get('filename', ''))[0] not in existing]
    
    print(f"Total: {len(entries)}, Processed: {len(existing)}, Pending: {len(pending)}")
    
    if not pending:
        print("Nothing to process.")
        return
    
    task_queue = Queue()
    for entry in pending:
        task_queue.put((entry, fieldnames))
    
    num_workers = CONFIG["num_workers"]
    for _ in range(num_workers):
        task_queue.put(None)
    
    cuda_devices = CONFIG["cuda_devices"]
    
    workers = []
    for i in range(num_workers):
        cuda_id = cuda_devices[i % len(cuda_devices)]
        p = Process(target=worker, args=(i, task_queue, CONFIG, cuda_id))
        p.start()
        workers.append(p)
    
    for p in workers:
        p.join()
    
    print("All done!")


if __name__ == "__main__":
    main()

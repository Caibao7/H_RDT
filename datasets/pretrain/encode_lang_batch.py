import argparse
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Process, Queue

import h5py
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = os.environ.get("HRDT_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(PROJECT_ROOT)
from models.encoder.t5_encoder import T5Embedder


def collect_all_files(target_dir: str, force_overwrite: bool, output_root: str | None):
    all_files = []
    dataset_dirs = []
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path) and (item in ["extra", "test"] or item.startswith("part")):
            dataset_dirs.append(item_path)

    dataset_dirs.sort()
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        for task_name in os.listdir(dataset_dir):
            task_dir = os.path.join(dataset_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            hdf5_files = [f for f in os.listdir(task_dir) if f.endswith(".hdf5")]
            hdf5_files.sort(key=lambda x: int(x.split(".")[0]))
            for hdf5_file in hdf5_files:
                hdf5_path = os.path.join(task_dir, hdf5_file)
                file_index = hdf5_file.split(".")[0]
                if output_root is None:
                    pt_path = os.path.join(task_dir, f"{file_index}.pt")
                else:
                    relative_dir = os.path.relpath(task_dir, target_dir)
                    output_dir = os.path.join(output_root, relative_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    pt_path = os.path.join(output_dir, f"{file_index}.pt")
                if force_overwrite or not os.path.exists(pt_path):
                    all_files.append(
                        {
                            "hdf5_path": hdf5_path,
                            "pt_path": pt_path,
                            "task_name": task_name,
                            "dataset_name": dataset_name,
                            "file_index": file_index,
                        }
                    )
    return all_files


def extract_instructions(hdf5_path: str):
    instructions = []
    with h5py.File(hdf5_path, "r") as f:
        if "llm_description" in f.attrs:
            instruction = f.attrs["llm_description"]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            instructions.append(str(instruction))

        if "llm_description2" in f.attrs:
            instruction2 = f.attrs["llm_description2"]
            if isinstance(instruction2, bytes):
                instruction2 = instruction2.decode("utf-8")
            instructions.append(str(instruction2))
    return instructions


def worker_process(process_id, gpu_id, file_list, progress_queue, args):
    try:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        with open(args.config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)

        text_embedder = T5Embedder(
            from_pretrained=args.model_path,
            model_max_length=args.max_length or config["dataset"].get("tokenizer_max_length", 128),
            device=device,
            local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

        print(f"Process {process_id} (GPU {gpu_id}) starting to process {len(file_list)} files")
        processed_count = 0
        failed_count = 0

        for file_info in file_list:
            try:
                hdf5_path = file_info["hdf5_path"]
                pt_path = file_info["pt_path"]

                if os.path.exists(pt_path) and not args.force_overwrite:
                    processed_count += 1
                    progress_queue.put(("processed", process_id))
                    continue

                instructions = extract_instructions(hdf5_path)
                if not instructions:
                    print(f"Process {process_id}: Warning: no instructions found in {hdf5_path}")
                    failed_count += 1
                    progress_queue.put(("failed", process_id))
                    continue

                tokenized_res = tokenizer(
                    instructions,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=text_embedder.model_max_length,
                )
                tokens = tokenized_res["input_ids"].to(device)
                attn_mask = tokenized_res["attention_mask"].to(device)

                with torch.no_grad():
                    text_embeds = text_encoder(input_ids=tokens, attention_mask=attn_mask)["last_hidden_state"].detach().cpu()
                attn_mask_cpu = attn_mask.cpu().bool()

                payload = {
                    "instruction": instructions[0],
                    "embeddings": text_embeds[0][attn_mask_cpu[0]],
                    "task_name": file_info["task_name"],
                    "dataset": file_info["dataset_name"],
                    "file_index": file_info["file_index"],
                    "text_model_name_or_path": args.model_path,
                    "embedding_dim": int(text_embeds.shape[-1]),
                }
                if len(instructions) > 1:
                    payload["instruction2"] = instructions[1]
                    payload["embeddings2"] = text_embeds[1][attn_mask_cpu[1]]

                torch.save(payload, pt_path)
                processed_count += 1
                progress_queue.put(("processed", process_id))

            except Exception as e:
                print(f"Process {process_id}: Error processing {file_info['hdf5_path']}: {e}")
                failed_count += 1
                progress_queue.put(("failed", process_id))

        print(f"Process {process_id} (GPU {gpu_id}) completed: successful {processed_count}, failed {failed_count}")
        progress_queue.put(("done", process_id, processed_count, failed_count))
    except Exception as e:
        print(f"Process {process_id} encountered serious error: {e}")
        progress_queue.put(("error", process_id, str(e)))


def progress_monitor(total_files, progress_queue, num_processes):
    processed = 0
    failed = 0
    finished_processes = 0
    pbar = tqdm(total=total_files, desc="Encoding EgoDex language")
    while finished_processes < num_processes:
        try:
            msg = progress_queue.get(timeout=1)
            if msg[0] == "processed":
                processed += 1
                pbar.update(1)
            elif msg[0] == "failed":
                failed += 1
                pbar.update(1)
            elif msg[0] == "done":
                finished_processes += 1
                print(f"\nProcess {msg[1]} completed: successful {msg[2]}, failed {msg[3]}")
            elif msg[0] == "error":
                finished_processes += 1
                print(f"\nProcess {msg[1]} error: {msg[2]}")
        except Exception:
            continue
    pbar.close()
    return processed, failed


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-encode EgoDex language annotations into .pt files.")
    parser.add_argument("--data_root", type=str, default=os.environ.get("EGODEX_DATA_ROOT", "/share/hongzhe/datasets/egodex"))
    parser.add_argument("--model_path", type=str, default="google-t5/t5-small")
    parser.add_argument("--config_path", type=str, default=os.environ.get("HRDT_CONFIG_PATH", os.path.join(PROJECT_ROOT, "configs/mini_vla_egodex.yaml")))
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", 1)))
    parser.add_argument("--processes_per_gpu", type=int, default=int(os.environ.get("PROCESSES_PER_GPU", 1)))
    parser.add_argument("--local_files_only", action="store_true", default=False)
    parser.add_argument("--force_overwrite", action="store_true", default=False)
    parser.add_argument("--test_mode", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    total_processes = max(args.num_gpus * args.processes_per_gpu, 1)

    print("Collecting EgoDex files for language encoding...")
    all_files = collect_all_files(args.data_root, force_overwrite=args.force_overwrite, output_root=args.output_root)
    if args.test_mode:
        all_files = all_files[:10]

    if not all_files:
        print("No files need processing.")
        return

    print(f"Model: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root or '[next to hdf5]'}")
    print(f"Files to process: {len(all_files)}")
    print(f"Processes: {total_processes} ({args.num_gpus} GPU x {args.processes_per_gpu} proc/GPU)")

    files_per_process = len(all_files) // total_processes
    file_lists = []
    for i in range(total_processes):
        start_idx = i * files_per_process
        end_idx = len(all_files) if i == total_processes - 1 else start_idx + files_per_process
        file_lists.append(all_files[start_idx:end_idx])

    progress_queue = Queue()
    monitor_process = Process(target=progress_monitor, args=(len(all_files), progress_queue, total_processes))
    monitor_process.start()

    processes = []
    for i in range(total_processes):
        gpu_id = i // args.processes_per_gpu
        process = Process(target=worker_process, args=(i, gpu_id, file_lists[i], progress_queue, args))
        process.start()
        processes.append(process)
        time.sleep(0.5)

    for process in processes:
        process.join()
    monitor_process.join()
    print("All processes completed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

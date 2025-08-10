from pathlib import Path

def get_model_paths(root_dir="./checkpoints/nano_clip/logs"):
    root_dir = Path(root_dir)
    ckpt_paths = []

    for fold_dir in sorted(root_dir.glob("nano_clip_fold*")):
        target_dir = fold_dir / "version_0" / "checkpoints"
        if target_dir.exists():
            for ckpt_file in target_dir.glob("*.ckpt"):
                ckpt_paths.append(str(ckpt_file.as_posix()))  

    return ckpt_paths

import os


def compute_trainer_artifact_dir(args, *, root: str) -> str:
    if (model_id := args.trainer_model_id) is None:
        return root
    return os.path.join(root, model_id)


def compute_trainer_artifact_name(args, *, name: str) -> str:
    if (model_id := args.trainer_model_id) is None:
        return name
    return f"{model_id}_{name}"

"""Output-path resolution shared by local and HOPE training entrypoints."""

import os


def resolve_checkpoint_save_path(config, dataset, checkpoint_name):
    checkpoint_root = os.path.abspath(
        os.fspath(config.get('checkpoint_root', './myckpt'))
    )
    return os.path.join(checkpoint_root, dataset, checkpoint_name)

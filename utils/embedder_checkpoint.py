import os


DEFAULT_EMBEDDER_PATH = os.path.join('pretrained', 'embedder.pt')


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve_embedder_path(checkpoint_path=None, required=True):
    if checkpoint_path:
        resolved_path = checkpoint_path
        source = '--embedder-path'
    else:
        resolved_path = os.path.join(repo_root(), DEFAULT_EMBEDDER_PATH)
        source = 'default'

    if not os.path.isabs(resolved_path):
        resolved_path = os.path.abspath(resolved_path)

    if required and not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            'Embedder checkpoint not found at %s (%s). Place the file at %s '
            'or pass --embedder-path with the correct path.'
            % (resolved_path, source, os.path.join(repo_root(), DEFAULT_EMBEDDER_PATH))
        )

    return resolved_path

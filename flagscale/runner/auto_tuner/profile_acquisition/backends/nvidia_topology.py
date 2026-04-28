import re
from dataclasses import dataclass

TOPOLOGY_COMMAND = ["nvidia-smi", "topo", "-m"]
AUTO_PAIR_CLASSES = "auto"
SUPPORTED_PAIR_CLASSES = ("pix", "sys")
GPU_TOKEN_PREFIX = "GPU"
ANSI_SEQUENCE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
KNOWN_TOPOLOGY_LINKS = frozenset(("x", "pix", "pxb", "phb", "node", "sys"))
NVLINK_TOKEN_PREFIX = "nv"


@dataclass(frozen=True)
class GpuPair:
    link_class: str
    devices: tuple[int, int]


def select_representative_pairs(topology_output, requested):
    matrix = _parse_topology_matrix(topology_output)
    requested_classes = _parse_requested_classes(requested)
    selected = _find_representative_pairs(matrix, requested_classes)
    missing = [link_class for link_class in requested_classes if link_class not in selected]
    if missing:
        raise ValueError(f"Missing representative GPU pair for classes: {', '.join(missing)}")
    return {link_class: selected[link_class] for link_class in requested_classes}


def _parse_requested_classes(requested):
    if requested == AUTO_PAIR_CLASSES:
        return SUPPORTED_PAIR_CLASSES
    if not requested or not requested.strip():
        raise ValueError("GPU pair classes must be a comma-separated list or auto.")
    classes = tuple(item.strip().lower() for item in requested.split(",") if item.strip())
    if not classes:
        raise ValueError("GPU pair classes must include at least one class.")
    unsupported = [link_class for link_class in classes if link_class not in SUPPORTED_PAIR_CLASSES]
    if unsupported:
        supported = ", ".join((*SUPPORTED_PAIR_CLASSES, AUTO_PAIR_CLASSES))
        unsupported_names = ", ".join(unsupported)
        message = f"Unsupported GPU pair classes: {unsupported_names}. Supported: {supported}"
        raise ValueError(message)
    return classes


def _parse_topology_matrix(topology_output):
    lines = _clean_topology_lines(topology_output)
    if not lines:
        raise ValueError("NVIDIA topology output is empty.")
    gpu_columns = _parse_gpu_columns(lines[0])
    matrix = {}
    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        row_gpu = _parse_gpu_token(tokens[0])
        if row_gpu is None:
            continue
        link_tokens = _parse_link_tokens(row_gpu, gpu_columns, tokens)
        matrix[row_gpu] = _parse_peer_links(row_gpu, gpu_columns, link_tokens)
    if not matrix:
        raise ValueError("NVIDIA topology output contains no GPU rows.")
    return matrix


def _clean_topology_lines(topology_output):
    lines = []
    for line in topology_output.splitlines():
        clean_line = _strip_ansi_control_sequences(line)
        if clean_line.strip():
            lines.append(clean_line)
    return lines


def _parse_gpu_columns(header_line):
    gpu_columns = []
    for token in header_line.split():
        gpu_id = _parse_gpu_token(token)
        if gpu_id is not None:
            gpu_columns.append(gpu_id)
    if not gpu_columns:
        raise ValueError("NVIDIA topology output contains no GPU columns.")
    return tuple(gpu_columns)


def _parse_link_tokens(row_gpu, gpu_columns, row_tokens):
    link_count = len(gpu_columns)
    link_tokens = row_tokens[1 : link_count + 1]
    if len(link_tokens) < link_count:
        raise ValueError(f"NVIDIA topology row for GPU{row_gpu} is incomplete.")
    invalid = [token for token in link_tokens if not _is_topology_link_token(token)]
    if invalid:
        raise ValueError(
            f"NVIDIA topology row for GPU{row_gpu} has invalid link token: {invalid[0]}"
        )
    return link_tokens


def _parse_peer_links(row_gpu, gpu_columns, link_tokens):
    peers = {}
    for peer_gpu, link_class in zip(gpu_columns, link_tokens):
        if peer_gpu == row_gpu:
            continue
        peers[peer_gpu] = link_class.lower()
    return peers


def _find_representative_pairs(matrix, requested_classes):
    selected = {}
    for row_gpu in sorted(matrix):
        for peer_gpu in sorted(matrix[row_gpu]):
            if peer_gpu <= row_gpu:
                continue
            link_class = matrix[row_gpu][peer_gpu]
            if link_class in requested_classes and link_class not in selected:
                selected[link_class] = GpuPair(link_class, (row_gpu, peer_gpu))
    return selected


def _parse_gpu_token(token):
    clean_token = _strip_ansi_control_sequences(token)
    if not clean_token.startswith(GPU_TOKEN_PREFIX):
        return None
    gpu_id = clean_token[len(GPU_TOKEN_PREFIX) :]
    if not gpu_id.isdigit():
        return None
    return int(gpu_id)


def _is_topology_link_token(token):
    link_class = token.lower()
    if link_class in KNOWN_TOPOLOGY_LINKS:
        return True
    if not link_class.startswith(NVLINK_TOKEN_PREFIX):
        return False
    suffix = link_class[len(NVLINK_TOKEN_PREFIX) :]
    return bool(suffix) and suffix.isalnum()


def _strip_ansi_control_sequences(value):
    without_ansi = ANSI_SEQUENCE_RE.sub("", value)
    return CONTROL_CHAR_RE.sub("", without_ansi)

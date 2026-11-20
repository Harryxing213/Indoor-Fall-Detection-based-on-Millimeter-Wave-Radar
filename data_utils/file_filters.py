from pathlib import Path


def is_multi_person(path):
    name = Path(path).name
    return '双人' in name or '三人' in name


def is_single_person(path):
    return '一人' in Path(path).name and not is_multi_person(path)

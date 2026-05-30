import pytest
import numpy as np
from pathlib import Path
from src.dataset.converter import convert_annotation, VALID_CLASSES


def test_valid_classes_mapping():
    """Test that class mapping is correct."""
    assert VALID_CLASSES[1] == 0   # pedestrian
    assert VALID_CLASSES[10] == 9  # motor
    assert 0 not in VALID_CLASSES  # ignored region excluded
    assert 11 not in VALID_CLASSES # others excluded


def test_convert_annotation_basic(tmp_path):
    """Test basic annotation conversion."""
    ann_file = tmp_path / "test.txt"
    ann_file.write_text("100,200,50,80,1,4,0,0\n")  # car

    lines = convert_annotation(ann_file, img_w=1920, img_h=1080)

    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "3"  # car = class 3
    assert 0 <= float(parts[1]) <= 1  # cx normalized
    assert 0 <= float(parts[2]) <= 1  # cy normalized
    assert 0 <= float(parts[3]) <= 1  # w normalized
    assert 0 <= float(parts[4]) <= 1  # h normalized


def test_convert_annotation_ignores_score_zero(tmp_path):
    """Test that ignored regions are skipped."""
    ann_file = tmp_path / "test.txt"
    ann_file.write_text("100,200,50,80,0,4,0,0\n")  # score=0 ignored

    lines = convert_annotation(ann_file, img_w=1920, img_h=1080)
    assert len(lines) == 0


def test_convert_annotation_ignores_zero_size(tmp_path):
    """Test that zero size boxes are skipped."""
    ann_file = tmp_path / "test.txt"
    ann_file.write_text("100,200,0,0,1,4,0,0\n")  # w=0 h=0

    lines = convert_annotation(ann_file, img_w=1920, img_h=1080)
    assert len(lines) == 0


def test_convert_annotation_clamps_values(tmp_path):
    """Test that values are clamped to [0,1]."""
    ann_file = tmp_path / "test.txt"
    ann_file.write_text("1900,1000,100,100,1,4,0,0\n")  # near edge

    lines = convert_annotation(ann_file, img_w=1920, img_h=1080)
    assert len(lines) == 1
    parts = lines[0].split()
    for val in parts[1:]:
        assert 0 <= float(val) <= 1


def test_convert_annotation_empty_file(tmp_path):
    """Test empty annotation file."""
    ann_file = tmp_path / "test.txt"
    ann_file.write_text("")

    lines = convert_annotation(ann_file, img_w=1920, img_h=1080)
    assert len(lines) == 0
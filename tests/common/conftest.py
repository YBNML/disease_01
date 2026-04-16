import json
import numpy as np
import cv2
import pytest
from pathlib import Path


@pytest.fixture
def sample_json_with_polygon(tmp_path):
    """Write a synthetic AI Hub-style JSON label with polygon, return its path."""
    payload = {
        "Info": {
            "IMAGE_FILE_NM": "TEST_000001",
            "RSOLTN": "(1920,1080)",
            "CMRA_INFO": "samsung",
            "LCINFO": "F02",
            "IMAGE_OBTAIN_PLACE_TY": "노지",
            "GRWH_STEP_CODE": "6",
            "OCPRD": "08-05",
            "SPCIES_NM": "온주밀감",
        },
        "Annotations": {
            "ANTN_ID": "1",
            "ANTN_TY": "polygon",
            "OBJECT_CLASS_CODE": "감귤_궤양병",
            "ANTN_PT": "[100|200|300|200],[100|50|100|200]",
        },
        "Environment": {
            "SOLRAD_QY": "34.7",
            "AFR": "0",
            "TP": "29.6",
            "HD": "64.8",
            "SOIL_MITR": "74.9",
        },
    }
    p = tmp_path / "TEST_000001.json"
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return p


@pytest.fixture
def sample_json_no_polygon(tmp_path):
    """JSON label with null annotations (no polygon)."""
    payload = {
        "Info": {
            "IMAGE_FILE_NM": "TEST_000002",
            "RSOLTN": "(1920,1080)",
            "CMRA_INFO": "Xiaomi",
            "LCINFO": "F19",
            "IMAGE_OBTAIN_PLACE_TY": "온실",
            "GRWH_STEP_CODE": "7",
            "OCPRD": "10-05",
            "SPCIES_NM": "온주밀감",
        },
        "Annotations": {
            "ANTN_ID": None,
            "ANTN_TY": None,
            "OBJECT_CLASS_CODE": "감귤_정상",
            "ANTN_PT": None,
        },
        "Environment": {
            "SOLRAD_QY": "20.1",
            "AFR": "0",
            "TP": "25.0",
            "HD": "60.0",
            "SOIL_MITR": "70.0",
        },
    }
    p = tmp_path / "TEST_000002.json"
    p.write_text(json.dumps(payload, ensure_ascii=False))
    return p


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """Create a minimal AI Hub-style directory tree with 2 normal + 2 canker images,
    where 1 of each has a polygon. Returns the DS root path."""
    root = tmp_path / "database"
    for split, split_dir in [("1.Training", "TS1.감귤"), ("2.Validation", "VS1.감귤")]:
        split_lbl = {"1.Training": "TL1.감귤", "2.Validation": "VL1.감귤"}[split]
        for cls_name, cls_code, dbyhs in [("열매_정상", "감귤_정상", "00"),
                                           ("열매_궤양병", "감귤_궤양병", "01")]:
            img_dir = root / split / "원천데이터" / split_dir / cls_name
            lbl_dir = root / split / "라벨링데이터" / split_lbl / cls_name
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            for i in range(2):
                stem = f"HF01_{dbyhs}FT_{i:06d}"
                # 64x64 solid-color image
                img = np.full((64, 64, 3), 200, dtype=np.uint8)
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)

                # polygon only for the first sample of each class
                has_poly = (i == 0)
                payload = {
                    "Info": {
                        "IMAGE_FILE_NM": stem,
                        "RSOLTN": "(64,64)",
                        "CMRA_INFO": "samsung",
                        "LCINFO": "F02",
                        "IMAGE_OBTAIN_PLACE_TY": "노지",
                        "GRWH_STEP_CODE": "6",
                        "OCPRD": "08-05",
                        "SPCIES_NM": "온주밀감",
                    },
                    "Annotations": {
                        "ANTN_ID": "1" if has_poly else None,
                        "ANTN_TY": "polygon" if has_poly else None,
                        "OBJECT_CLASS_CODE": cls_code,
                        "ANTN_PT": "[10|50|50|10],[10|10|50|50]" if has_poly else None,
                    },
                    "Environment": {
                        "SOLRAD_QY": "30.0", "AFR": "0",
                        "TP": "25.0", "HD": "60.0", "SOIL_MITR": "70.0",
                    },
                }
                (lbl_dir / f"{stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False)
                )
    return root

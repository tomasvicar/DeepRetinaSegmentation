"""Adapter contracts: auxiliary labels must never become negative segmentations."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from deepretina.adapters.auxiliary import ADAPTERS
from deepretina.sources import Source


def image(root, name, pixels=None):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), 128, np.uint8) if pixels is None else pixels).save(path)


def text(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def run(root, selector):
    records = []

    def emit(stem, split, rgb, target, **kwargs):
        records.append({"stem": stem, "split": split, "image": rgb, "target": target, **kwargs})

    with Source(root) as source:
        ADAPTERS[selector](source, emit)
    return records


@pytest.mark.parametrize(
    "selector,filename",
    [
        ("diaretdb0", "diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images/image111.png"),
        ("diaretdb1", "diaretdb1_v_1_1/resources/images/ddb1_fundusimages/image049.png"),
        ("eophtha", "e_optha_EX/EX/patient1/abc.jpg"),
        ("heimed", "HEI-MED-master/DMED/patient.dcm.jpg"),
        ("rcrgbma", "RC-RGB-MA/Original/img-original (1).jpg"),
        ("ridb", "RIDB/IM000001_2.JPG"),
    ],
)
def test_image_only_collections(selector, filename, tmp_path):
    image(tmp_path, filename)
    rows = run(tmp_path, selector)
    assert len(rows) == 1
    assert rows[0]["target"] is None
    assert rows[0]["task"] == "images"
    assert rows[0]["split"] == "unsplit"
    if selector in {"diaretdb0", "diaretdb1"}:
        assert rows[0]["metadata"]["legacy_diagnosis"] == "healthy"


def test_erlangen_exact_visit_green_manual_mask_and_subject_group(tmp_path):
    stem = "Study_02_00061_03_L"
    image(tmp_path, f"Erlangen/FundusImages/{stem}_reg.png")
    mask = np.zeros((6, 8, 3), np.uint8)
    mask[2:4, 3:5, 1] = 255
    image(
        tmp_path,
        f"Erlangen/Segmentation_Blood_Vessels/Manually_corrected/{stem}_registered_bv_corrected.tif",
        mask,
    )
    image(tmp_path, "Erlangen/FundusImages/Study_02_00061_02_R_reg.png")
    rows = run(tmp_path, "erlangen")
    supervised = [r for r in rows if r["target"] is not None]
    assert len(supervised) == 1
    assert supervised[0]["target"].shape == (1, 12, 16)
    assert supervised[0]["target"].sum() == 16
    assert len({r["source_id"] for r in rows}) == 1
    assert sum(r["target"] is None for r in rows) == 1


def test_fire_registration_pair_keeps_original_points(tmp_path):
    for number in (1, 2):
        image(tmp_path, f"FIRE/Images/P01_{number}.jpg")
    text(tmp_path, "FIRE/Ground Truth/control_points_P01_1_2.txt", "1 2 3 4\n5 6 7 8\n")
    image(tmp_path, "FIRE/Masks/mask.png", np.ones((12, 16), np.uint8) * 255)
    rows = run(tmp_path, "fire")
    assert len(rows) == 2 and rows[0]["source_id"] == rows[1]["source_id"]
    assert rows[0]["metadata"]["control_points_xy_image1_xy_image2"][0] == [1, 2, 3, 4]
    assert rows[0]["fov"].all()


def test_idrid_localization_is_not_disc_segmentation(tmp_path):
    image(tmp_path, "IDRiD/C. Localization/1. Original Images/b. Testing Set/IDRiD_001.jpg")
    text(
        tmp_path,
        "IDRiD/C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv",
        "Image No,X,Y\nIDRiD_001,120,77\n",
    )
    row = run(tmp_path, "idrid")[0]
    assert row["split"] == "test" and row["target"] is None
    assert row["metadata"]["coordinates_original_xy"]["optic_disc_center_xy"] == [120, 77]


def test_inspire_measurements_and_missing_labels_error(tmp_path):
    image(tmp_path, "INSPIRE-AVR/org/image1.jpg")
    with pytest.raises(ValueError, match="GT/GT.txt"):
        run(tmp_path, "inspireavr")
    text(tmp_path, "INSPIRE-AVR/GT/GT.txt", "AVR obs1 obs2\nimage1.jpg 0.7 0.71\n")
    row = run(tmp_path, "inspireavr")[0]
    assert row["metadata"]["avr_observer_1_2"] == [0.7, 0.71]
    assert row["target"] is None


def test_messidor_grading_table(tmp_path, monkeypatch):
    image(tmp_path, "MESSIDOR/Base11/patient.tif")
    text(tmp_path, "MESSIDOR/Annotation_Base11.xls", "fixture parsed by mocked xlrd")
    sheet = SimpleNamespace(nrows=2, row_values=lambda i: ["patient.tif", "date", 2, 1])
    workbook = SimpleNamespace(sheet_by_index=lambda i: sheet)
    monkeypatch.setitem(
        sys.modules, "xlrd", SimpleNamespace(open_workbook=lambda **kwargs: workbook)
    )
    row = run(tmp_path, "messidor")[0]
    assert row["metadata"]["retinopathy_grade"] == 2
    assert row["metadata"]["macular_edema_risk"] == 1
    assert row["target"] is None


def test_review_keeps_observer_edges_without_filling_vessels(tmp_path):
    image(tmp_path, "Review/KPIS/KPIS001.bmp")
    text(
        tmp_path,
        "Review/Observer Marking for KPIS 1.txt",
        "1 1 2 " + " ".join(str(i) for i in range(12)) + "\n",
    )
    row = run(tmp_path, "review")[0]
    assert row["target"] is None
    assert row["metadata"]["vessel_edge_measurements_original"][0]["observer_right_left_xy"][2] == [
        8,
        9,
        10,
        11,
    ]


def test_roc_preserves_official_test_and_sparse_radius(tmp_path):
    image(tmp_path, "ROC/ROCtraining/image1_training.jpg")
    image(tmp_path, "ROC/images/image1_test.jpg")
    text(
        tmp_path,
        "ROC/ROCtraining/annotations-consensus-ma-only.xml",
        '<set><annotations-per-image imagename="image1_training.jpg"><annotation><mark x="3" y="4"><radius>5</radius></mark><lesion>microaneurysm</lesion><probability>1</probability></annotation></annotations-per-image></set>',
    )
    rows = run(tmp_path, "roc")
    assert {r["split"] for r in rows} == {"train", "test"}
    train = next(r for r in rows if r["split"] == "train")
    assert train["metadata"]["lesion_marks_original_xy"][0]["radius"] == 5
    assert all(r["target"] is None for r in rows)


def test_vicavr_keeps_caliber_class_and_coordinates(tmp_path):
    image(tmp_path, "VICAVR/images/original/Image1.jpg")
    text(tmp_path, "VICAVR/expert1.txt", "# Image,vessel,x,y,caliber,type\n1,0,409,337,8.88897,2\n")
    row = run(tmp_path, "vicavr")[0]
    measurement = row["metadata"]["point_measurements_original_xy"][0]
    assert (
        measurement["class"] == 2 and measurement["x"] == 409 and measurement["caliber"] == 8.88897
    )
    assert row["target"] is None


def test_ubmi_optional_dense_mask_and_missing_data(tmp_path):
    with pytest.raises(ValueError, match="provide original"):
        run(tmp_path, "ubmi")
    image(tmp_path, "Gacr_01_017_01_L_data/Gacr_01_017_01_L.jpg")
    mask = np.zeros((12, 16), np.uint8)
    mask[4:8, 6:10] = 255
    image(tmp_path, "Gacr_01_017_01_L_data/ImageAnalysis/VesselsSeg/tree_whole.png", mask)
    row = run(tmp_path, "ubmi")[0]
    assert row["target"] is None and row["task"] == "images"
    assert (
        row["metadata"]["annotation_resources"][0]["kind"]
        == "existing_vessel_segmentation_for_inference"
    )
    assert row["fov"].all()


@pytest.mark.parametrize("selector", list(ADAPTERS))
def test_every_adapter_rejects_empty_layout(selector, tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, selector)


def test_diaretdb_lesion_categories_are_not_collapsed(tmp_path):
    image(tmp_path, "diaretdb1/resources/images/ddb1_fundusimages/image001.png")
    for category in ["hardexudates", "softexudates", "hemorrhages", "redsmalldots"]:
        image(tmp_path, f"diaretdb1/resources/images/ddb1_groundtruth/{category}/image001.png")
    row = run(tmp_path, "diaretdb1")[0]
    assert len(row["metadata"]["annotation_resources"]) == 4
    assert row["target"] is None


def test_duplicate_transport_prefixes_do_not_duplicate_auxiliary_annotations(tmp_path):
    image(tmp_path, "FIRE/FIRE/Images/P01_1.jpg")
    text(tmp_path, "FIRE/FIRE/Ground Truth/control_points_P01_1_2.txt", "1 2 3 4\n")
    text(tmp_path, "FIRE/Ground Truth/control_points_P01_1_2.txt", "1 2 3 4\n")
    row = run(tmp_path, "fire")[0]
    assert len(row["metadata"]["annotation_resources"]) == 1

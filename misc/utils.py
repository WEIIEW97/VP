from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl


@dataclass
class CameraCalib:
    name: str
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    model_type: str


def parse_camera_payload(cell: object) -> dict | None:
    if not isinstance(cell, str):
        return None

    text = cell.strip()
    start = text.find("{'")
    if start < 0:
        start = text.find('{"')
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        payload = ast.literal_eval(text[start : end + 1])
    except (SyntaxError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None
    if "cam_intrinsic" not in payload or "cam_distcoeffs" not in payload:
        return None
    return payload


def payload_to_calib(serial: str, payload: dict) -> CameraCalib:
    return CameraCalib(
        name=serial.strip(),
        K=np.asarray(payload["cam_intrinsic"], dtype=np.float64).reshape(3, 3),
        dist=np.asarray(payload["cam_distcoeffs"], dtype=np.float64),
        width=int(payload["resolution_width"]),
        height=int(payload["resolution_height"]),
        model_type=str(payload.get("model_type", "")).upper(),
    )


def row_payload(row: tuple[object, ...]) -> dict | None:
    payload = next((p for p in map(parse_camera_payload, row[1:]) if p), None)
    if payload is not None:
        return payload

    joined = ",".join("" if cell is None else str(cell) for cell in row[1:])
    return parse_camera_payload(joined)


def sheet_calibs(ws) -> list[CameraCalib]:
    calibs: list[CameraCalib] = []
    for row in ws.iter_rows(values_only=True):
        serial = row[0]
        if not isinstance(serial, str) or not serial.strip():
            continue

        payload = row_payload(row)
        if payload is not None:
            calibs.append(payload_to_calib(serial, payload))
    return calibs


def load_intrinsics(path: Path) -> tuple[list[CameraCalib], str]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    candidates = [(ws.title, sheet_calibs(ws)) for ws in wb.worksheets]
    sheet, calibs = max(candidates, key=lambda item: len(item[1]))
    if not calibs:
        raise ValueError(f"No camera intrinsics parsed from {path}")
    return calibs, sheet

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "experiments.db"
ARTIFACTS_DIR = BACKEND_DIR / "experiment_artifacts"


def get_connection() -> sqlite3.Connection:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    connection = get_connection()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                filename TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_checkpoint TEXT,
                medsam_checkpoint TEXT,
                device TEXT,
                inference_seconds REAL,
                medsam_score REAL,
                dice REAL,
                iou REAL,
                precision REAL,
                recall REAL,
                bbox_json TEXT,
                config_json TEXT NOT NULL,
                input_path TEXT,
                prediction_path TEXT,
                error TEXT
            )
        """)
        connection.commit()
    finally:
        connection.close()


def create_experiment(
    *,
    filename: str,
    model: str,
    prompt_checkpoint: str,
    medsam_checkpoint: str,
    device: str,
    config: dict[str, Any],
) -> str:
    experiment_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()

    connection = get_connection()
    try:
        connection.execute("""
            INSERT INTO experiments (
                id, created_at, status, filename, model,
                prompt_checkpoint, medsam_checkpoint, device,
                config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            created_at,
            "running",
            filename,
            model,
            prompt_checkpoint,
            medsam_checkpoint,
            device,
            json.dumps(config),
        ))
        connection.commit()
    finally:
        connection.close()

    return experiment_id


def complete_experiment(
    experiment_id: str,
    *,
    input_bytes: bytes,
    mask_png_bytes: bytes,
    inference_seconds: float,
    medsam_score: float | None,
    bbox: list[float],
    metrics: dict[str, Any] | None = None,
) -> None:
    artifact_dir = ARTIFACTS_DIR / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_path = artifact_dir / "input_image"
    prediction_path = artifact_dir / "predicted_mask.png"
    input_path.write_bytes(input_bytes)
    prediction_path.write_bytes(mask_png_bytes)

    connection = get_connection()
    try:
        connection.execute("""
            UPDATE experiments
            SET status = ?,
                inference_seconds = ?,
                medsam_score = ?,
                dice = ?,
                iou = ?,
                precision = ?,
                recall = ?,
                bbox_json = ?,
                input_path = ?,
                prediction_path = ?,
                error = NULL
            WHERE id = ?
        """, (
            "completed",
            inference_seconds,
            medsam_score,
            metrics.get("dice") if metrics else None,
            metrics.get("iou") if metrics else None,
            metrics.get("precision") if metrics else None,
            metrics.get("recall") if metrics else None,
            json.dumps(bbox),
            str(input_path),
            str(prediction_path),
            experiment_id,
        ))
        connection.commit()
    finally:
        connection.close()


def update_experiment_metrics(
    experiment_id: str,
    metrics: dict[str, Any],
) -> None:
    """Persist computed segmentation metrics for an existing experiment."""
    connection = get_connection()
    try:
        cursor = connection.execute("""
            UPDATE experiments
            SET dice = ?, iou = ?, precision = ?, recall = ?
            WHERE id = ?
        """, (
            metrics.get("dice"),
            metrics.get("iou"),
            metrics.get("precision"),
            metrics.get("recall"),
            experiment_id,
        ))
        if cursor.rowcount == 0:
            raise ValueError(f"Experiment not found: {experiment_id}")
        connection.commit()
    finally:
        connection.close()


def fail_experiment(experiment_id: str, error: str) -> None:
    connection = get_connection()
    try:
        connection.execute("""
            UPDATE experiments
            SET status = ?, error = ?
            WHERE id = ?
        """, ("failed", error[:2000], experiment_id))
        connection.commit()
    finally:
        connection.close()


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["bbox"] = json.loads(result.pop("bbox_json") or "[]")
    result["config"] = json.loads(result.pop("config_json") or "{}")
    result["input_url"] = (
        f"/api/experiments/{result['id']}/input"
        if result.get("input_path") else None
    )
    result["prediction_url"] = (
        f"/api/experiments/{result['id']}/prediction"
        if result.get("prediction_path") else None
    )
    result.pop("input_path", None)
    result.pop("prediction_path", None)
    return result


def list_experiments(limit: int = 100) -> list[dict[str, Any]]:
    connection = get_connection()
    try:
        rows = connection.execute("""
            SELECT * FROM experiments
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    finally:
        connection.close()
    return [_serialize(row) for row in rows]


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    finally:
        connection.close()
    return _serialize(row) if row else None


def get_artifact_path(
    experiment_id: str,
    artifact: str,
) -> Path | None:
    if artifact not in {"input", "prediction"}:
        return None

    experiment = get_experiment(experiment_id)
    if not experiment:
        return None

    path = (
        ARTIFACTS_DIR / experiment_id / "input_image"
        if artifact == "input"
        else ARTIFACTS_DIR / experiment_id / "predicted_mask.png"
    )
    return path if path.is_file() else None


initialize_database()

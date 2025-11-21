import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
_env_spec_path = os.getenv("TAG_SPEC_PATH")
if _env_spec_path:
    DEFAULT_SPEC_PATH = Path(_env_spec_path)
    if not DEFAULT_SPEC_PATH.is_absolute():
        DEFAULT_SPEC_PATH = (Path.cwd() / DEFAULT_SPEC_PATH).resolve()
else:
    DEFAULT_SPEC_PATH = MODULE_DIR / "data" / "tag_spec.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _coerce_types(record: Dict) -> Dict:
    record = dict(record)
    # Normalize expected keys
    if "module_length" not in record and "module_height" in record:
        record["module_length"] = record.get("module_height")
    if "module_width" in record:
        record["module_width"] = record.get("module_width")
    # Ensure commissioning_date is a simple string (YYYY-MM-DD)
    comm = record.get("commissioning_date")
    if pd.notna(comm):
        record["commissioning_date"] = str(comm)
    else:
        record["commissioning_date"] = None
    return record


@lru_cache(maxsize=1)
def _load_spec_df() -> pd.DataFrame:
    if not DEFAULT_SPEC_PATH.exists():
        logger.error("Tag specification file not found: %s", DEFAULT_SPEC_PATH)
        return pd.DataFrame()
    df = pd.read_csv(DEFAULT_SPEC_PATH)
    df = _normalize_columns(df)
    return df


def get_tag_specification(topic: str) -> Optional[Dict]:
    df = _load_spec_df()
    if df.empty:
        return None

    topic_col = "tag"
    if topic_col not in df.columns:
        logger.error("Specification file missing required 'tag' column")
        return None

    matches = df[df[topic_col] == topic]
    if matches.empty:
        logger.warning("No specification found for topic '%s'", topic)
        return None

    record = matches.iloc[0].to_dict()
    record = _coerce_types(record)
    return record


def list_available_tags() -> List[Dict[str, str]]:
    df = _load_spec_df()
    if df.empty:
        return []

    if "tag" not in df.columns:
        logger.error("Specification file missing required 'tag' column")
        return []

    tag_type_col = "tag_type" if "tag_type" in df.columns else None
    records = []
    for _, row in df.iterrows():
        entry = {"tag": str(row["tag"])}
        if tag_type_col:
            entry["tag_type"] = str(row[tag_type_col])
        records.append(entry)

    return records

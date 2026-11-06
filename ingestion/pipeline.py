"""
ingestion/pipeline.py
=====================
Orchestrates the complete offline ingestion and data preparation pipeline:
1. Loads raw cricket log with LangChain TextLoader
2. Parses records deterministically using compiled regex
3. Normalizes, cleans, and extracts optional commentary metadata
4. Validates each record against domain constraints
5. Produces a detailed Data Quality & Ingestion Report
"""

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .loader import load_cricket_log
from .parser import parse_log_line
from .cleaner import clean_and_enrich_record, CleanedCricketRecord
from .validator import validate_record, ValidationReport


def run_ingestion_pipeline(
    file_path: Path | None = None
) -> Tuple[List[CleanedCricketRecord], ValidationReport, Dict[str, Any]]:
    """
    Executes the end-to-end ingestion and validation pipeline.
    
    Returns:
        tuple: (
            valid_records: list of validated CleanedCricketRecord instances,
            report: ValidationReport containing validation metrics,
            stats: summary statistics dictionary (event counts, run totals, etc.)
        )
    """
    # 1. Load document lines with LangChain TextLoader
    documents, file_metadata = load_cricket_log(file_path)

    valid_records: List[CleanedCricketRecord] = []
    unparsed_lines: List[Tuple[int, str]] = []
    validation_report = ValidationReport(total_processed=len(documents))

    # 2. Parse, clean, and validate line-by-line
    for doc in documents:
        line_num = doc.metadata.get("line_number", 0)
        raw_text = doc.page_content

        raw_record = parse_log_line(raw_text, line_number=line_num)
        if not raw_record:
            unparsed_lines.append((line_num, raw_text))
            validation_report.invalid_count += 1
            validation_report.errors_by_record[f"LINE_{line_num}"] = ["Regex parsing failed"]
            continue

        # Clean & normalize
        cleaned_record = clean_and_enrich_record(raw_record)

        # Validate
        errors = validate_record(cleaned_record)
        if errors:
            validation_report.invalid_count += 1
            validation_report.errors_by_record[cleaned_record.event_id] = errors
        else:
            validation_report.valid_count += 1
            valid_records.append(cleaned_record)

    # 3. Aggregate data quality statistics
    total_runs = sum(r.runs for r in valid_records)
    event_counts = dict(Counter(r.event for r in valid_records))
    status_counts = dict(Counter(r.status for r in valid_records))
    match_counts = dict(Counter(r.match_id for r in valid_records))
    team_counts = dict(Counter(r.team for r in valid_records))
    player_counts = dict(Counter(r.player for r in valid_records))

    stats = {
        "file_metadata": file_metadata,
        "total_lines_read": len(documents),
        "unparsed_count": len(unparsed_lines),
        "valid_count": validation_report.valid_count,
        "invalid_count": validation_report.invalid_count,
        "total_runs": total_runs,
        "event_distribution": event_counts,
        "status_distribution": status_counts,
        "match_distribution": match_counts,
        "team_distribution": team_counts,
        "unique_players": len(player_counts),
        "optional_fields_present": {
            "player_role_count": sum(1 for r in valid_records if r.player_role is not None),
            "venue_count": sum(1 for r in valid_records if r.venue is not None),
            "weather_count": sum(1 for r in valid_records if r.weather_condition is not None),
        }
    }

    return valid_records, validation_report, stats

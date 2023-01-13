import csv
import logging
import os


def save_results(results: dict[str, float], out_dir: str) -> None:
    results_filepath = os.path.join(out_dir, "results.csv")
    logging.info("Saving results to %s.", results_filepath)
    with open(results_filepath, mode="w", newline="", encoding="utf8") as csv_file:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)

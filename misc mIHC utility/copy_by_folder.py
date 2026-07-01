import csv
import os
import shutil
import time
from datetime import datetime


SOURCE_ROOT = r"V:\\"
DESTINATION_ROOT = r"Z:\Multiplex_IHC_studies\Isaac_Youm\D8_Panel_StudySlides\Slides"
#r"Z:\Multiplex_IHC_studies\Isaac_Youm\KateByrne_D8Panel_Practice_PDACctrl\Slides"
FILENAME_PREFIX = "IY"
OLDEST_DATE = "2026-06-23"
NEWEST_DATE = None
SIMLINK_INSTEAD = 0
DEBUG_CSV_NAME = "prefix_folder_match_debug.csv"
DATE_FORMAT = "%Y-%m-%d"
MAX_RETRIES = 10
RETRY_WAIT_SECONDS = 60


def get_date(folder_name):
    try:
        return datetime.strptime(folder_name, DATE_FORMAT).date()
    except ValueError:
        return None


def listdir_with_retries(path):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return os.listdir(path)
        except FileNotFoundError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_WAIT_SECONDS)


def get_slides_root():
    if os.path.basename(os.path.normpath(DESTINATION_ROOT)).lower() == "slides":
        return DESTINATION_ROOT
    return os.path.join(DESTINATION_ROOT, "Slides")


def get_source_files(oldest_date, newest_date, rows):
    source_files = []
    for dir_name in sorted(listdir_with_retries(SOURCE_ROOT)):
        dir_path = os.path.join(SOURCE_ROOT, dir_name)
        if not os.path.isdir(dir_path):
            continue
        folder_date = get_date(dir_name)
        if folder_date is None:
            continue
        if oldest_date and folder_date < oldest_date:
            continue
        if newest_date and folder_date > newest_date:
            continue
        try:
            files = listdir_with_retries(dir_path)
        except FileNotFoundError:
            rows.append(["missing_folder", "", dir_path, "Folder disappeared before scan"])
            continue
        for file_name in files:
            source_path = os.path.join(dir_path, file_name)
            if os.path.isfile(source_path) and file_name.startswith(FILENAME_PREFIX):
                source_files.append([file_name, source_path])
    return source_files


def choose_slide_folders(source_files, slides_root):
    potentials = []
    for file_name, _source_path in source_files:
        parts = file_name.split("_")
        if len(parts) > 2 and parts[2] not in potentials:
            potentials.append(parts[2])

    if not potentials:
        return []

    while True:
        good = []
        print("No slide folders were found in the destination.")
        print(f"If you continue, this script will create the Slides folder if needed: {slides_root}")
        print("Potential slide names found from prefix-matching source files:")
        for i, potential in enumerate(potentials):
            print(i, ":", potential)
        while True:
            try:
                inp = int(input("include slide number:"))
                good.append(potentials[inp])
            except Exception:
                break
        good = list(dict.fromkeys(good))
        print("This script will create these slide folders and then route files into them:")
        print(good)
        if input("use this list? (y)").strip().lower() == "y":
            return good


def main():
    oldest_date = get_date(OLDEST_DATE) if OLDEST_DATE else None
    newest_date = get_date(NEWEST_DATE) if NEWEST_DATE else None
    slides_root = get_slides_root()
    rows = []
    source_files = get_source_files(oldest_date, newest_date, rows)

    if os.path.isdir(slides_root):
        outfolders = [
            name
            for name in os.listdir(slides_root)
            if os.path.isdir(os.path.join(slides_root, name))
        ]
    else:
        outfolders = []

    if not outfolders:
        outfolders = choose_slide_folders(source_files, slides_root)
        os.makedirs(slides_root, exist_ok=True)
        for outfolder in outfolders:
            os.makedirs(os.path.join(slides_root, outfolder), exist_ok=True)

    movelist = []
    for file_name, source_path in source_files:
        matches = [
            outfolder
            for outfolder in outfolders
            if file_name.startswith(FILENAME_PREFIX) and outfolder.lower() in file_name.lower()
        ]
        if len(matches) != 1:
            debug_text = "No matching destination folder" if not matches else "Multiple matching destination folders: " + ", ".join(matches)
            rows.append(["debug", file_name, source_path, debug_text])
            continue
        dest_path = os.path.join(slides_root, matches[0], file_name)
        if os.path.lexists(dest_path):
            rows.append(["duplicate", file_name, source_path, f"Exact filename already exists: {dest_path}"])
            continue
        movelist.append([source_path, dest_path, file_name])

    for source_path, dest_path, file_name in movelist:
        if os.path.lexists(dest_path):
            rows.append(["duplicate", file_name, source_path, f"Exact filename already exists: {dest_path}"])
            continue
        try:
            if SIMLINK_INSTEAD:
                os.symlink(source_path, dest_path)
                rows.append(["symlinked", file_name, source_path, dest_path])
            else:
                shutil.copy2(source_path, dest_path)
                rows.append(["copied", file_name, source_path, dest_path])
        except Exception as exc:
            rows.append(["error", file_name, source_path, f"{type(exc).__name__}: {exc}"])

    os.makedirs(slides_root, exist_ok=True)
    csv_path = os.path.join(slides_root, DEBUG_CSV_NAME)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", "filename", "source", "destination_or_debug"])
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print(f"Queued {len(movelist)} files")


if __name__ == "__main__":
    main()

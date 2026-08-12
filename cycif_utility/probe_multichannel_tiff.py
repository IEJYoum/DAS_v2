import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tiff


DEFAULT_GLOB = r"\\accsmb.ohsu.edu\cedar-scmeth\ChinData\CycIF_FB3_whole-section\*.tif"


def pick_path(path_glob):
    hits = sorted(glob.glob(path_glob))
    if not hits:
        raise FileNotFoundError(f"No TIFF matched: {path_glob}")
    if len(hits) > 1:
        print(f"Matched {len(hits)} TIFFs; using first:")
    print(hits[0])
    return hits[0]


def parse_ome_channel_names(ome_xml):
    if not ome_xml:
        return []
    try:
        root = ET.fromstring(ome_xml)
    except Exception:
        return []
    channel_names = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Channel":
            name = elem.attrib.get("Name")
            if name is None or str(name).strip() == "":
                name = elem.attrib.get("ID", "")
            channel_names.append(str(name))
    return channel_names


def channel_axis_info(series):
    axes = str(getattr(series, "axes", ""))
    shape = tuple(int(x) for x in getattr(series, "shape", ()))
    if "C" not in axes:
        return axes, shape, None, None
    c_index = axes.index("C")
    c_count = shape[c_index]
    return axes, shape, c_index, c_count


def try_single_channel_read(path, series, c_index, c_count):
    result = {
        "ok": False,
        "method": None,
        "shape": None,
        "dtype": None,
        "message": "",
    }

    if c_index is None or c_count is None:
        result["message"] = "No C axis in series"
        return result

    try:
        page = series.pages[0]
        arr = page.asarray()
        result["ok"] = True
        result["method"] = "series.pages[0].asarray()"
        result["shape"] = tuple(int(x) for x in arr.shape)
        result["dtype"] = str(arr.dtype)
        result["message"] = "Read first stored page only; confirms page-level access."
        return result
    except Exception as e:
        result["message"] = f"Page read failed: {e}"

    try:
        import zarr

        store = series.aszarr()
        z = zarr.open(store, mode="r")
        sel = [slice(None)] * len(z.shape)
        sel[c_index] = 0
        arr = np.asarray(z[tuple(sel)])
        result["ok"] = True
        result["method"] = "series.aszarr() + zarr slice on C axis"
        result["shape"] = tuple(int(x) for x in arr.shape)
        result["dtype"] = str(arr.dtype)
        result["message"] = "Read one channel lazily through zarr-backed slice."
        return result
    except Exception as e:
        result["message"] += f" | zarr slice failed: {e}"

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_GLOB, help="TIFF path or glob")
    args = ap.parse_args()

    path = pick_path(args.path)
    print("")
    print("=== Probe ===")
    print("path:", path)
    print("size_gb:", round(os.path.getsize(path) / (1024 ** 3), 2))

    with tiff.TiffFile(path) as tf:
        print("series_count:", len(tf.series))
        print("page_count:", len(tf.pages))
        if len(tf.series) == 0:
            print("No series found")
            return

        series = tf.series[0]
        axes, shape, c_index, c_count = channel_axis_info(series)
        print("series0_axes:", axes)
        print("series0_shape:", shape)
        print("series0_dtype:", getattr(series, "dtype", None))
        print("series0_kind:", type(series).__name__)

        names = parse_ome_channel_names(getattr(tf, "ome_metadata", None))
        if names:
            print("channel_names_count:", len(names))
            for i, name in enumerate(names[:128]):
                print(f"channel[{i}] = {name}")
        else:
            print("channel_names_count: 0")

        if c_index is None:
            print("No C axis detected in series 0.")
        else:
            print("channel_axis_index:", c_index)
            print("channel_count:", c_count)

        read_result = try_single_channel_read(path, series, c_index, c_count)
        print("")
        print("=== Single-channel read test ===")
        print("ok:", read_result["ok"])
        print("method:", read_result["method"])
        print("shape:", read_result["shape"])
        print("dtype:", read_result["dtype"])
        print("message:", read_result["message"])

    print("")
    print("Expected nuclear channel name: DAPI1")
    print("Next step if probe looks good: build a wrapper that maps OME channel names to DAS-style synthetic channel entries and processes one channel at a time.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Probe failed:", e)
        sys.exit(1)

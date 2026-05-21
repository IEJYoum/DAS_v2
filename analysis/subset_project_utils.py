import json


def normalize_subset_values(values, sort_key=None):
    out = []
    seen = set()
    items = list(values or [])
    i = 0
    while i < len(items):
        text = str(items[i]).strip()
        if text != "" and text not in seen:
            seen.add(text)
            out.append(text)
        i += 1
    if callable(sort_key):
        return sorted(out, key=sort_key)
    return sorted(out)


def sanitize_subset_token(text):
    text = str(text).strip()
    repl = [
        ("<", "_"),
        (">", "_"),
        (":", "."),
        ('"', "_"),
        ("/", "_"),
        ("\\", "_"),
        ("|", "_"),
        ("?", "_"),
        ("*", "_"),
    ]
    i = 0
    while i < len(repl):
        text = text.replace(repl[i][0], repl[i][1])
        i += 1
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip(" .")
    return text or "blank"


def subset_mode_text(mode):
    low = str(mode).strip().lower()
    if low in ["0", "include"]:
        return "include"
    return "exclude"


def subset_folder_label(column, mode, values, sort_key=None):
    col_token = sanitize_subset_token(column)
    normed = normalize_subset_values(values, sort_key=sort_key)
    value_tokens = []
    i = 0
    while i < len(normed):
        value_tokens.append(sanitize_subset_token(normed[i]))
        i += 1
    joined = "+".join(value_tokens) if len(value_tokens) > 0 else "blank"
    mode_text = subset_mode_text(mode)
    if mode_text == "include" and len(value_tokens) == 1:
        return f"{col_token}={value_tokens[0]}"
    if mode_text == "include":
        return f"{col_token}__include={joined}"
    return f"{col_token}__exclude={joined}"


def subset_values_storage_text(values, sort_key=None):
    return json.dumps(normalize_subset_values(values, sort_key=sort_key), ensure_ascii=False)


def parse_saved_subset_values(text, sort_key=None):
    raw = str(text or "").strip()
    if raw == "":
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return normalize_subset_values(parsed, sort_key=sort_key)
        except Exception:
            pass
    if "||" in raw:
        return normalize_subset_values(raw.split("||"), sort_key=sort_key)
    if "\n" in raw:
        return normalize_subset_values(raw.splitlines(), sort_key=sort_key)
    return normalize_subset_values([raw], sort_key=sort_key)


def subset_project_definition_from_config(config, sort_key=None):
    source = dict(config or {})
    column = str(source.get("subset_column", "")).strip()
    mode = str(source.get("subset_mode", "")).strip().lower()
    values = parse_saved_subset_values(source.get("subset_values_json", ""), sort_key=sort_key)
    if column == "" or mode not in ["include", "exclude"] or len(values) == 0:
        return None
    return {
        "column": column,
        "mode": mode,
        "values": values,
        "label": str(source.get("subset_label", "")).strip(),
        "parent_folder": str(source.get("subset_parent_folder", "")).strip(),
    }


def subset_definition_matches_config(config, column, values, mode="include", sort_key=None):
    definition = subset_project_definition_from_config(config, sort_key=sort_key)
    if not isinstance(definition, dict):
        return False
    if str(definition.get("column", "")).strip() != str(column).strip():
        return False
    if str(definition.get("mode", "")).strip().lower() != subset_mode_text(mode):
        return False
    left = normalize_subset_values(definition.get("values", []), sort_key=sort_key)
    right = normalize_subset_values(values, sort_key=sort_key)
    return left == right


def next_subset_project_name(child_names):
    next_n = 1
    names = list(child_names or [])
    i = 0
    while i < len(names):
        text = str(names[i]).strip()
        low = text.lower()
        if low.startswith("subset_"):
            suffix = text[len("subset_"):].strip()
            if suffix.isdigit():
                try:
                    next_n = max(next_n, int(suffix) + 1)
                except Exception:
                    pass
        i += 1
    return f"subset_{next_n:03d}"

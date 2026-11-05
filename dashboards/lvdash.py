"""Small builders for AI/BI (Lakeview) dashboard JSON, shared by the dashboard generators.

build_dashboard.py (benchmarks) and build_cost_dashboard.py (costs) assemble their
`*.lvdash.json` from these, so widget shapes live in one place. Every helper returns plain
dicts; key order is kept stable so regenerated JSON diffs cleanly.
"""
from __future__ import annotations

# A widget query field: (name, SQL expression). The name doubles as the encoding fieldName.
Field = tuple[str, str]


def dataset(name: str, display: str, sql: str, parameters: list[tuple[str, str, str, str]] | None = None,
            columns: list[dict] | None = None) -> dict:
    """A SQL dataset. parameters: (keyword, display name, data type, default); columns: measures."""
    ds = {"name": name, "displayName": display, "queryLines": [line + "\n" for line in sql.split("\n")]}
    if parameters:
        ds["parameters"] = [
            {"displayName": label, "keyword": kw, "dataType": dtype,
             "defaultSelection": {"values": {"dataType": dtype, "values": [{"value": default}]}}}
            for kw, label, dtype, default in parameters]
    if columns:
        ds["columns"] = columns
    return ds


def measure(display: str, expression: str, description: str = "") -> dict:
    """A dataset-level measure, referenced from widgets as MEASURE(`display`)."""
    m = {"displayName": display, "expression": expression}
    if description:
        m["description"] = description
    return m


def query(name: str, ds: str, fields: list[Field], disaggregated: bool = True,
          filters: list[str] | None = None) -> dict:
    q: dict = {"datasetName": ds, "disaggregated": disaggregated,
               "fields": [{"name": n, "expression": e} for n, e in fields]}
    if filters:
        q["filters"] = [{"expression": f} for f in filters]
    return {"name": name, "query": q}


def cols(*names: str) -> list[Field]:
    return [(n, f"`{n}`") for n in names]


def agg(fn: str, column: str) -> Field:
    """An aggregated field, e.g. agg("sum", "list_usd") -> ("sum(list_usd)", "SUM(`list_usd`)")."""
    return f"{fn}({column})", f"{fn.upper()}(`{column}`)"


def daily(column: str) -> Field:
    return f"daily({column})", f'DATE_TRUNC("DAY", `{column}`)'


def weekly(column: str) -> Field:
    return f"weekly({column})", f'DATE_TRUNC("WEEK", `{column}`)'


def measure_field(display: str) -> Field:
    return f"measure({display})", f"MEASURE(`{display}`)"


def text(name: str, md: str, x: int, y: int, w: int, h: int) -> dict:
    return {"widget": {"name": name, "textbox_spec": md}, "position": {"x": x, "y": y, "width": w, "height": h}}


def counter(name: str, ds: str, column: str, title: str, x: int, y: int, w: int = 1, h: int = 3,
            fmt: dict | None = None, field: Field | None = None, period: Field | None = None,
            filters: list[str] | None = None) -> dict:
    """A KPI tile: SUM(column) by default, or any aggregated `field`; optional sparkline `period`."""
    name_, expr = field or (f"sum({column})", f"SUM(`{column}`)")
    fields = [(name_, expr)] + ([period] if period else [])
    value: dict = {"fieldName": name_, "displayName": title}
    if fmt:
        value["format"] = fmt
    encodings: dict = {"value": value}
    if period:
        encodings["period"] = {"fieldName": period[0]}
    return {
        "widget": {
            "name": name,
            "queries": [query("main_query", ds, fields, disaggregated=False, filters=filters)],
            "spec": {"version": 2, "widgetType": "counter", "encodings": encodings,
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def chart(name: str, kind: str, ds: str, title: str, x_enc: dict, y_enc: dict, color: dict | None,
          pos: tuple[int, int, int, int], extra: dict | None = None, fields: list[Field] | None = None,
          filters: list[str] | None = None) -> dict:
    """A v3 chart. Without `fields` it reads raw columns; pass aggregated fields for grouped data."""
    disaggregated = fields is None
    if fields is None:
        names = [x_enc["fieldName"], y_enc["fieldName"]] + ([color["fieldName"]] if color else [])
        fields = cols(*names)
    encodings = {"x": x_enc, "y": y_enc}
    if color:
        encodings["color"] = color
    spec = {"version": 3, "widgetType": kind, "encodings": encodings, "frame": {"title": title, "showTitle": True}}
    spec.update(extra or {})
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, fields, disaggregated, filters)], "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def pie(name: str, ds: str, title: str, angle: Field, color: str, pos: tuple[int, int, int, int],
        fmt: dict | None = None, filters: list[str] | None = None) -> dict:
    angle_enc: dict = {"fieldName": angle[0], "scale": {"type": "quantitative"}, "displayName": title}
    if fmt:
        angle_enc["format"] = fmt
    spec = {"version": 3, "widgetType": "pie",
            "encodings": {"angle": angle_enc, "color": cat(color, color.replace("_", " ").capitalize()),
                          "label": {"show": True}},
            "frame": {"title": title, "showTitle": True}}
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, [angle, (color, f"`{color}`")], False, filters)],
                       "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def cat(field: str, label: str) -> dict:
    return {"fieldName": field, "scale": {"type": "categorical"}, "displayName": label}


def num(field: str, label: str, lo: float | None = 0, hi: float | None = 1, fmt: dict | None = None) -> dict:
    scale: dict = {"type": "quantitative"}
    if lo is not None and hi is not None:
        scale["domain"] = {"min": lo, "max": hi}
    enc = {"fieldName": field, "scale": scale, "displayName": label}
    if fmt:
        enc["format"] = fmt
    return enc


def temporal(field: str, label: str) -> dict:
    return {"fieldName": field, "scale": {"type": "temporal"}, "displayName": label}


def filter_widget(name: str, field: str, title: str, datasets: list[str], pos: tuple[int, int, int, int],
                  multi: bool = False) -> dict:
    queries = [query(f"f_{ds}", ds, cols(field), disaggregated=False) for ds in datasets]
    x, y, w, h = pos
    return {
        "widget": {
            "name": name,
            "queries": queries,
            "spec": {"version": 2, "widgetType": "filter-multi-select" if multi else "filter-single-select",
                     "encodings": {"fields": [{"fieldName": field, "displayName": title, "queryName": q["name"]}
                                              for q in queries]},
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def date_range_filter(name: str, field: str, title: str, datasets: list[str], pos: tuple[int, int, int, int],
                      default_min: str = "now-30d/d", default_max: str = "now/d") -> dict:
    """A field-bound date-range filter over every listed dataset (each must have `field`)."""
    queries = [query(f"d_{ds}", ds, cols(field), disaggregated=False) for ds in datasets]
    x, y, w, h = pos
    return {
        "widget": {
            "name": name,
            "queries": queries,
            "spec": {"version": 2, "widgetType": "filter-date-range-picker",
                     "encodings": {"fields": [{"fieldName": field, "displayName": title, "queryName": q["name"]}
                                              for q in queries]},
                     "selection": {"defaultSelection": {"range": {
                         "dataType": "DATE", "min": {"value": default_min}, "max": {"value": default_max}}}},
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def parameter_widget(name: str, keyword: str, title: str, ds: str, pos: tuple[int, int, int, int]) -> dict:
    q = {"name": f"parameter_{ds}_{keyword}",
         "query": {"datasetName": ds, "parameters": [{"name": keyword, "keyword": keyword}], "disaggregated": False}}
    x, y, w, h = pos
    return {
        "widget": {
            "name": name,
            "queries": [q],
            "spec": {"version": 2, "widgetType": "filter-single-select",
                     "encodings": {"fields": [{"parameterName": keyword, "queryName": q["name"]}]},
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def number_fmt(places: int) -> dict:
    return {"type": "number-plain", "abbreviation": "none", "decimalPlaces": {"type": "exact", "places": places}}


def usd_fmt(places: int = 2) -> dict:
    return {"type": "number-currency", "currencyCode": "USD", "abbreviation": "compact",
            "decimalPlaces": {"type": "max", "places": places}}


def compact_fmt(places: int = 1) -> dict:
    return {"type": "number", "abbreviation": "compact", "decimalPlaces": {"type": "max", "places": places}}


def pct_fmt(places: int = 1) -> dict:
    return {"type": "number-percent", "decimalPlaces": {"type": "max", "places": places}}


def rule(field: str, value: str, color: str) -> dict:
    return {"condition": {"operator": "=", "fieldName": field, "operand": {"type": "data-value", "value": value}},
            "backgroundColor": color}


def table(name: str, ds: str, title: str, columns: list[dict], pos: tuple[int, int, int, int], freeze: int = 0) -> dict:
    hidden = [c.pop("_also", []) for c in columns]
    fields = [c["fieldName"] for c in columns] + [f for extra in hidden for f in extra]
    spec = {"version": 2, "widgetType": "table", "encodings": {"columns": columns},
            "frame": {"title": title, "showTitle": True}, "data": {"queryName": "main_query"}}
    if freeze:
        spec["freezeUpToColumnNumber"] = freeze
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, cols(*dict.fromkeys(fields)))], "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def col(field: str, label: str, places: int | None = None, rules: list | None = None, template: str | None = None,
        also: list[str] | None = None, fmt: dict | None = None) -> dict:
    c: dict = {"fieldName": field, "displayName": label}
    if places is not None:
        c["format"] = number_fmt(places)
    if fmt:
        c["format"] = fmt
    if template:
        c["formatTemplate"] = template
    if rules:
        c["style"] = {"type": "basic", "rules": rules}
    if also:
        c["_also"] = also
    return c


def pivot(name: str, ds: str, title: str, rows: list[str], cells: list[tuple[Field, str, dict | None]],
          pos: tuple[int, int, int, int], filters: list[str] | None = None) -> dict:
    """Rows = dimensions (raw columns), cells = (aggregated field, label, format) summed per row."""
    fields = cols(*rows) + [f for f, _, _ in cells]
    cell_fields = []
    for (fname, _), label, fmt in cells:
        cf: dict = {"fieldName": fname, "cellType": "text", "displayName": label}
        if fmt:
            cf["format"] = fmt
        cell_fields.append(cf)
    spec = {"version": 3, "widgetType": "pivot",
            "encodings": {"rows": [{"fieldName": r, "displayName": r.replace("_", " ").capitalize()} for r in rows],
                          "cell": {"type": "multi-cell", "fields": cell_fields}},
            "frame": {"title": title, "showTitle": True}}
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, fields, False, filters)], "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}

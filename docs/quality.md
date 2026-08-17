# Code Quality Report

This page is generated automatically by `dev/generate_docs.py`.

## Test Coverage

**Status:** ✅ Pass (100.00% Coverage)

## Static Type Checking (Mypy)

**Status:** ❌ Fail (34 Errors)

## Linting (Ruff)

**Status:** ❌ Fail (11 issues)

## sklearn Estimator Checks

**Status:** ✅ Pass (5/5 estimators passed)

| Estimator | Status | Detail |
|:--|:--:|:--|
| RadiomicsFeatureRemover | ✅ Pass | OK |
| RadiomicsPrepTransformer | ✅ Pass | OK |
| WGCNAReducer | ✅ Pass | OK (failed safely on synthetic network size limits). |
| PCAReducer | ✅ Pass | OK |
| SparsePCAReducer | ✅ Pass | OK |

## Package Build

**Status:** ✅ Pass (- Built eigenradiomics-0.1.0-py3-none-any.whl)

```text
eigenradiomics/batch_effects.py:148: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/plotting.py:1341: error: Incompatible types in assignment (expression has type "None", variable has type "Axes")  [assignment]
eigenradiomics/plotting.py:1343: error: Item "None" of "Axes | None" has no attribute "figure"  [union-attr]
eigenradiomics/plotting.py:1343: error: Incompatible types in assignment (expression has type "Figure | SubFigure | Any", variable has type "Figure")  [assignment]
eigenradiomics/plotting.py:1352: error: Incompatible types in assignment (expression has type "None", variable has type "Axes")  [assignment]
eigenradiomics/plotting.py:1371: error: Item "None" of "Axes | None" has no attribute "plot"  [union-attr]
eigenradiomics/plotting.py:1389: error: Item "None" of "Axes | None" has no attribute "add_patch"  [union-attr]
eigenradiomics/plotting.py:1395: error: Item "None" of "Axes | None" has no attribute "add_patch"  [union-attr]
eigenradiomics/plotting.py:1412: error: Item "None" of "Axes | None" has no attribute "add_patch"  [union-attr]
eigenradiomics/plotting.py:1413: error: Item "None" of "Axes | None" has no attribute "add_patch"  [union-attr]
eigenradiomics/plotting.py:1416: error: Item "None" of "Axes | None" has no attribute "set_xlim"  [union-attr]
eigenradiomics/plotting.py:1417: error: Item "None" of "Axes | None" has no attribute "set_ylim"  [union-attr]
eigenradiomics/plotting.py:1422: error: Item "None" of "Axes | None" has no attribute "set_yticks"  [union-attr]
eigenradiomics/plotting.py:1423: error: Item "None" of "Axes | None" has no attribute "set_yticklabels"  [union-attr]
eigenradiomics/plotting.py:1424: error: Item "None" of "Axes | None" has no attribute "twinx"  [union-attr]
eigenradiomics/plotting.py:1425: error: Item "None" of "Axes | None" has no attribute "get_ylim"  [union-attr]
eigenradiomics/plotting.py:1429: error: Item "None" of "Axes | None" has no attribute "set_yticks"  [union-attr]
eigenradiomics/plotting.py:1430: error: Item "None" of "Axes | None" has no attribute "twinx"  [union-attr]
eigenradiomics/plotting.py:1433: error: Item "None" of "Axes | None" has no attribute "set_xticks"  [union-attr]
eigenradiomics/plotting.py:1434: error: Item "None" of "Axes | None" has no attribute "set_xticklabels"  [union-attr]
eigenradiomics/plotting.py:1436: error: Item "None" of "Axes | None" has no attribute "spines"  [union-attr]
eigenradiomics/plotting.py:1525: error: Item "None" of "Axes | None" has no attribute "axis"  [union-attr]
eigenradiomics/plotting.py:1527: error: Item "None" of "Axes | None" has no attribute "legend"  [union-attr]
eigenradiomics/reproducibility.py:558: error: List item 0 has incompatible type "Sequence[Axes] | Any"; expected "Axes"  [list-item]
eigenradiomics/reproducibility.py:560: error: Value of type "Sequence[Axes] | None" is not indexable  [index]
eigenradiomics/reproducibility.py:560: error: Incompatible types in assignment (expression has type "Figure | SubFigure | Any", variable has type "Figure")  [assignment]
eigenradiomics/reproducibility.py:561: error: Argument 1 to "len" has incompatible type "Sequence[Axes] | None"; expected "Sized"  [arg-type]
eigenradiomics/reproducibility.py:562: error: Argument 1 to "len" has incompatible type "Sequence[Axes] | None"; expected "Sized"  [arg-type]
eigenradiomics/reproducibility.py:572: error: Argument 1 to "enumerate" has incompatible type "Sequence[Axes] | None"; expected "Iterable[Axes]"  [arg-type]
eigenradiomics/reproducibility.py:587: error: Argument "bins" to "hist" of "Axes" has incompatible type "ndarray[tuple[int], dtype[float64 | Any]]"; expected "int | Sequence[float] | str | None"  [arg-type]
eigenradiomics/reproducibility.py:663: error: Argument 1 to "len" has incompatible type "Sequence[Axes] | None"; expected "Sized"  [arg-type]
eigenradiomics/reproducibility.py:664: error: Value of type "Sequence[Axes] | None" is not indexable  [index]
eigenradiomics/reproducibility.py:783: error: Incompatible types in assignment (expression has type "dict[str, Any] | None", variable has type "dict[str, Any]")  [assignment]
eigenradiomics/reproducibility.py:923: error: Item "None" of "Legend | None" has no attribute "get_window_extent"  [union-attr]
Found 34 errors in 3 files (checked 29 source files)
```

```text
[
  {
    "cell": null,
    "code": "SIM113",
    "end_location": {
      "column": 24,
      "row": 904
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/reproducibility.py",
    "fix": null,
    "location": {
      "column": 9,
      "row": 904
    },
    "message": "Use `enumerate()` for index variable `letter_idx` in `for` loop",
    "noqa_row": 904,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/enumerate-for-loop"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 103,
      "row": 407
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_analysis.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 407
    },
    "message": "Line too long (102 > 99)",
    "noqa_row": 407,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 104,
      "row": 416
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_analysis.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 416
    },
    "message": "Line too long (103 > 99)",
    "noqa_row": 416,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 116,
      "row": 427
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_analysis.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 427
    },
    "message": "Line too long (115 > 99)",
    "noqa_row": 427,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 101,
      "row": 467
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_reproducibility.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 467
    },
    "message": "Line too long (100 > 99)",
    "noqa_row": 467,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 103,
      "row": 469
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_reproducibility.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 469
    },
    "message": "Line too long (102 > 99)",
    "noqa_row": 469,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 106,
      "row": 497
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_reproducibility.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 497
    },
    "message": "Line too long (105 > 99)",
    "noqa_row": 497,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "I001",
    "end_location": {
      "column": 55,
      "row": 160
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_utils.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "        import matplotlib.pyplot as plt\n\n        from eigenradiomics._utils import _save_figure\n",
          "end_location": {
            "column": 1,
            "row": 161
          },
          "location": {
            "column": 1,
            "row": 159
          }
        }
      ],
      "message": "Organize imports"
    },
    "location": {
      "column": 9,
      "row": 159
    },
    "message": "Import block is un-sorted or un-formatted",
    "noqa_row": 159,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unsorted-imports"
  },
  {
    "cell": null,
    "code": "I001",
    "end_location": {
      "column": 55,
      "row": 179
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_utils.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "        import matplotlib.pyplot as plt\n\n        from eigenradiomics._utils import _save_figure\n",
          "end_location": {
            "column": 1,
            "row": 180
          },
          "location": {
            "column": 1,
            "row": 178
          }
        }
      ],
      "message": "Organize imports"
    },
    "location": {
      "column": 9,
      "row": 178
    },
    "message": "Import block is un-sorted or un-formatted",
    "noqa_row": 178,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unsorted-imports"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 104,
      "row": 1112
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_wgcna_reducer.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1112
    },
    "message": "Line too long (103 > 99)",
    "noqa_row": 1112,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 102,
      "row": 1154
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_wgcna_reducer.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1154
    },
    "message": "Line too long (101 > 99)",
    "noqa_row": 1154,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  }
]
```

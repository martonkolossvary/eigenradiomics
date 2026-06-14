# Code Quality Report

This page is generated automatically by `dev/generate_docs.py`.

## Test Coverage

**Status:** ✅ Pass (99.38% Coverage)

## Static Type Checking (Mypy)

**Status:** ❌ Fail (17 Errors)

## Linting (Ruff)

**Status:** ❌ Fail (36 issues)

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
eigenradiomics/reducers/_embeddings.py:91: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:163: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:224: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:269: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:324: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:383: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:437: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/reducers/_embeddings.py:489: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]"  [no-any-return]
eigenradiomics/plotting.py:943: error: Name "FeatureCatalog" is not defined  [name-defined]
eigenradiomics/feature_models.py:1024: error: Item "None" of "CorrPanel | None" has no attribute "data"  [union-attr]
eigenradiomics/feature_models.py:1133: error: Argument "strips" to "_draw_top_strips" has incompatible type "Sequence[Strip] | None"; expected "Sequence[Strip]"  [arg-type]
eigenradiomics/feature_models.py:1147: error: Argument "bars" to "_draw_bottom_bars" has incompatible type "Sequence[Bar] | None"; expected "Sequence[Bar]"  [arg-type]
eigenradiomics/feature_models.py:1158: error: Item "None" of "CorrPanel | None" has no attribute "data"  [union-attr]
eigenradiomics/feature_models.py:1163: error: Item "None" of "CorrPanel | None" has no attribute "cmap"  [union-attr]
eigenradiomics/feature_models.py:1164: error: Item "None" of "CorrPanel | None" has no attribute "vmin"  [union-attr]
eigenradiomics/feature_models.py:1165: error: Item "None" of "CorrPanel | None" has no attribute "vmax"  [union-attr]
eigenradiomics/feature_models.py:1187: error: Item "None" of "CorrPanel | None" has no attribute "label"  [union-attr]
Found 17 errors in 3 files (checked 29 source files)
```

```text
[
  {
    "cell": null,
    "code": "I001",
    "end_location": {
      "column": 62,
      "row": 35
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "    from pathlib import Path\n\n    import matplotlib.pyplot as plt\n\n    from eigenradiomics.plotting import Bar, CorrPanel, Strip\n",
          "end_location": {
            "column": 1,
            "row": 36
          },
          "location": {
            "column": 1,
            "row": 33
          }
        }
      ],
      "message": "Organize imports"
    },
    "location": {
      "column": 5,
      "row": 33
    },
    "message": "Import block is un-sorted or un-formatted",
    "noqa_row": 33,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unsorted-imports"
  },
  {
    "cell": null,
    "code": "SIM108",
    "end_location": {
      "column": 27,
      "row": 956
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": {
      "applicability": "unsafe",
      "edits": [
        {
          "content": "df = result.table.copy() if isinstance(result, FeatureAssociationResult) else result.copy()",
          "end_location": {
            "column": 27,
            "row": 956
          },
          "location": {
            "column": 5,
            "row": 953
          }
        }
      ],
      "message": "Replace `if`-`else`-block with `df = result.table.copy() if isinstance(result, FeatureAssociationResult) else result.copy()`"
    },
    "location": {
      "column": 5,
      "row": 953
    },
    "message": "Use ternary operator `df = result.table.copy() if isinstance(result, FeatureAssociationResult) else result.copy()` instead of `if`-`else`-block",
    "noqa_row": 953,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/if-else-block-instead-of-if-exp"
  },
  {
    "cell": null,
    "code": "B007",
    "end_location": {
      "column": 18,
      "row": 1071
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": {
      "applicability": "unsafe",
      "edits": [
        {
          "content": "_name",
          "end_location": {
            "column": 18,
            "row": 1071
          },
          "location": {
            "column": 14,
            "row": 1071
          }
        }
      ],
      "message": "Rename unused `name` to `_name`"
    },
    "location": {
      "column": 14,
      "row": 1071
    },
    "message": "Loop control variable `name` not used within loop body",
    "noqa_row": 1071,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unused-loop-control-variable"
  },
  {
    "cell": null,
    "code": "B007",
    "end_location": {
      "column": 16,
      "row": 1081
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": null,
    "location": {
      "column": 13,
      "row": 1081
    },
    "message": "Loop control variable `fam` not used within loop body",
    "noqa_row": 1081,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unused-loop-control-variable"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 102,
      "row": 1105
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1105
    },
    "message": "Line too long (101 > 99)",
    "noqa_row": 1105,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 107,
      "row": 1155
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/feature_models.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1155
    },
    "message": "Line too long (106 > 99)",
    "noqa_row": 1155,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "F821",
    "end_location": {
      "column": 28,
      "row": 943
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 14,
      "row": 943
    },
    "message": "Undefined name `FeatureCatalog`",
    "noqa_row": 943,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/undefined-name"
  },
  {
    "cell": null,
    "code": "F401",
    "end_location": {
      "column": 54,
      "row": 984
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 1,
            "row": 985
          },
          "location": {
            "column": 1,
            "row": 984
          }
        }
      ],
      "message": "Remove unused import: `eigenradiomics.catalog.FeatureCatalog`"
    },
    "location": {
      "column": 40,
      "row": 984
    },
    "message": "`eigenradiomics.catalog.FeatureCatalog` imported but unused",
    "noqa_row": 984,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unused-import"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 126,
      "row": 999
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 999
    },
    "message": "Line too long (125 > 99)",
    "noqa_row": 999,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 1066
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 1066
          },
          "location": {
            "column": 1,
            "row": 1066
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 1066
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 1066,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 130,
      "row": 1077
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1077
    },
    "message": "Line too long (129 > 99)",
    "noqa_row": 1077,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 128,
      "row": 1079
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1079
    },
    "message": "Line too long (127 > 99)",
    "noqa_row": 1079,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "I001",
    "end_location": {
      "column": 40,
      "row": 1114
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "    from matplotlib.lines import Line2D\n    from matplotlib.patches import Patch\n",
          "end_location": {
            "column": 1,
            "row": 1115
          },
          "location": {
            "column": 1,
            "row": 1113
          }
        }
      ],
      "message": "Organize imports"
    },
    "location": {
      "column": 5,
      "row": 1113
    },
    "message": "Import block is un-sorted or un-formatted",
    "noqa_row": 1113,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unsorted-imports"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 1118
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 1118
          },
          "location": {
            "column": 1,
            "row": 1118
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 1118
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 1118,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 102,
      "row": 1120
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1120
    },
    "message": "Line too long (101 > 99)",
    "noqa_row": 1120,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "E501",
    "end_location": {
      "column": 134,
      "row": 1123
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/plotting.py",
    "fix": null,
    "location": {
      "column": 100,
      "row": 1123
    },
    "message": "Line too long (133 > 99)",
    "noqa_row": 1123,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/line-too-long"
  },
  {
    "cell": null,
    "code": "I001",
    "end_location": {
      "column": 2,
      "row": 13
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/eigenradiomics/reducers/__init__.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "from eigenradiomics.reducers._base import BaseReducer\nfrom eigenradiomics.reducers._embeddings import (\n    IsomapReducer,\n    LLEReducer,\n    MDSReducer,\n    PaCMAPReducer,\n    SpectralReducer,\n    TriMAPReducer,\n    TSNEReducer,\n    UMAPReducer,\n)\nfrom eigenradiomics.reducers._pca import PCAReducer, SparsePCAReducer\nfrom eigenradiomics.reducers._wgcna import WGCNAReducer\n\n",
          "end_location": {
            "column": 1,
            "row": 15
          },
          "location": {
            "column": 1,
            "row": 1
          }
        }
      ],
      "message": "Organize imports"
    },
    "location": {
      "column": 1,
      "row": 1
    },
    "message": "Import block is un-sorted or un-formatted",
    "noqa_row": 1,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/unsorted-imports"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 34
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 34
          },
          "location": {
            "column": 1,
            "row": 34
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 34
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 34,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 38
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 38
          },
          "location": {
            "column": 1,
            "row": 38
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 38
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 38,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 46
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 46
          },
          "location": {
            "column": 1,
            "row": 46
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 46
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 46,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 50
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 50
          },
          "location": {
            "column": 1,
            "row": 50
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 50
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 50,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 58
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 58
          },
          "location": {
            "column": 1,
            "row": 58
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 58
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 58,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 62
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 62
          },
          "location": {
            "column": 1,
            "row": 62
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 62
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 62,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 70
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 70
          },
          "location": {
            "column": 1,
            "row": 70
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 70
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 70,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 73
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 73
          },
          "location": {
            "column": 1,
            "row": 73
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 73
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 73,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 81
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 81
          },
          "location": {
            "column": 1,
            "row": 81
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 81
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 81,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 84
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 84
          },
          "location": {
            "column": 1,
            "row": 84
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 84
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 84,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 93
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 93
          },
          "location": {
            "column": 1,
            "row": 93
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 93
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 93,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 96
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 96
          },
          "location": {
            "column": 1,
            "row": 96
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 96
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 96,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 110
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 110
          },
          "location": {
            "column": 1,
            "row": 110
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 110
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 110,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 116
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 116
          },
          "location": {
            "column": 1,
            "row": 116
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 116
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 116,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 131
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 131
          },
          "location": {
            "column": 1,
            "row": 131
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 131
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 131,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 137
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 137
          },
          "location": {
            "column": 1,
            "row": 137
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 137
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 137,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 5,
      "row": 148
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 5,
            "row": 148
          },
          "location": {
            "column": 1,
            "row": 148
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 148
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 148,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 156
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 156
          },
          "location": {
            "column": 1,
            "row": 156
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 156
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 156,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  },
  {
    "cell": null,
    "code": "W293",
    "end_location": {
      "column": 9,
      "row": 160
    },
    "filename": "/Users/mjk2/Library/CloudStorage/OneDrive-Personal/Python/eigenradiomics/tests/test_embeddings.py",
    "fix": {
      "applicability": "safe",
      "edits": [
        {
          "content": "",
          "end_location": {
            "column": 9,
            "row": 160
          },
          "location": {
            "column": 1,
            "row": 160
          }
        }
      ],
      "message": "Remove whitespace from blank line"
    },
    "location": {
      "column": 1,
      "row": 160
    },
    "message": "Blank line contains whitespace",
    "noqa_row": 160,
    "severity": "error",
    "url": "https://docs.astral.sh/ruff/rules/blank-line-with-whitespace"
  }
]
```

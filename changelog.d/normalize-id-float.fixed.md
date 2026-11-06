`normalize_id_column` now writes an integer-valued float ID as an integer (`1.0` becomes `"1"`), so it matches the same ID stored as an integer. A real decimal such as `1.5` stays `"1.5"`.

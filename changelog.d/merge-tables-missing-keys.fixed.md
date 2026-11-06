`merge_tables` no longer joins a row with a missing key to another row with a missing key. These rows go to `left_only` / `right_only` and do not count in `n_matched`.

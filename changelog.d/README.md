# Changelog Fragments

Drop news fragments here. Towncrier collects them into `docs/changelog.md` at release time.

## Naming convention

```
<issue_or_slug>.<type>.md
```

Where `<type>` is one of: `added`, `changed`, `fixed`, `removed`, `deprecated`.

### Examples

```
42.added.md          → references issue #42
sparse-error.fixed.md → freeform slug (no issue)
```

## Creating a fragment

```bash
poetry run towncrier create 42.added.md
# then edit the file with a one-line description
```

## Building the changelog (at release time)

```bash
poetry run towncrier build --version 0.2.0
```

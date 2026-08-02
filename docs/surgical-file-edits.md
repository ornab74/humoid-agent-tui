# Surgical file editing

Humoid can modify bounded regions of existing files without forcing the model to reconstruct and resend the entire document.

## Tools

- `append_file`: append content without reading or transmitting the complete existing file.
- `insert_text`: insert immediately before or after an exact anchor.
- `replace_text`: replace or delete exact text with strict match-count validation.
- `apply_file_edits`: apply several dependent mutations to one file atomically.
- `read_file_range`: inspect only the exact line range needed to establish a reliable anchor.
- `write_file`: retained for new files and intentional full regeneration.

All mutations use the existing unified-diff approval surface, create one undo checkpoint, and synchronize the active Project Perspective index.

## Safe concurrency

`read_file` and `read_file_range` return the complete file's SHA-256. Pass that value as `expected_sha256` to a mutation tool. Humoid refuses the edit when another process changed the file after inspection.

```json
{
  "path": "README.md",
  "expected_sha256": "...",
  "anchor": "## Installation\n",
  "position": "before",
  "content": "## Architecture\n...\n\n"
}
```

## Exact-match rules

Surgical tools default to one expected match. This prevents a short anchor such as `return value` from silently changing multiple unrelated functions.

Set `expected_matches` only after search or exact reading establishes the real count. `occurrence` is one-based; `-1` selects the final occurrence.

## Atomic multi-edit workflow

`apply_file_edits` evaluates every operation in memory before touching disk. If any anchor or replacement fails, no file content is written.

```json
{
  "path": "README.md",
  "expected_sha256": "...",
  "edits": [
    {
      "operation": "insert_before",
      "anchor": "## Summary",
      "content": "## New capability\n...\n\n"
    },
    {
      "operation": "replace",
      "old_text": "Supports 10 ideas",
      "content": "Supports 19 ideas"
    },
    {
      "operation": "append",
      "content": "\n## Changelog\nAdded surgical editing.\n"
    }
  ]
}
```

The resulting file is previewed as one diff, written once, and restored by one `undo_file_change` call.

## Recommended model behavior

1. Use Project Perspective search to locate the change surface.
2. Use `read_file_range` or a precise `read_file` only when exact anchor text is needed.
3. Preserve the returned SHA-256 as a precondition.
4. Prefer append, insert, replace, or atomic edits over whole-file reconstruction.
5. If an anchor is missing or ambiguous, inspect again rather than guessing or rewriting the whole file.
6. Run focused validation after the mutation.

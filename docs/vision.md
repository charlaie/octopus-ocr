# Vision Pipeline Notes

This project intentionally uses a UI-specific vision pipeline for the current Octopus transaction screen instead of generic document OCR. The screenshots are consistent, high-resolution app screens, so deterministic layout logic gives us more control than asking OCR to infer the whole page.

## Core Assumptions

- The reference screenshots are `1170x2532`. Pixel constants in `vision.py` are based on this layout and scaled by the current image width/height.
- The transaction list has a stable structure: icon on the left, payee/date text in the middle, amount on the right.
- Rows are detected from the icon column, not from OCR text. This avoids OCR deciding where a row starts or ends.
- Category is visual data, not text data. We classify it from the icon color/shape before OCR.
- OCR is only used for fields that must be read: payee, date/time, and amount.

## Why Icon/Color Detection

The Octopus app category icons are highly distinctive:

- Blue circular bus/train icon: `transport`
- Orange circular fork/knife icon: `eat and drink`
- Green circular shopping bag icon: `living and others`
- Small AAVS/Octopus logo: `top-up`

We use HSV color thresholds because these icon colors are saturated and stable across screenshots. HSV is less brittle than raw RGB for this kind of detection.

The AAVS/top-up logo also contains orange, so it can overlap the food icon hue range. To avoid classifying top-ups as food, full category badges must pass a larger area/radius check; small orange logo fragments remain top-up candidates.

## Important Layout Constants

These constants come from the current `1170x2532` screenshots:

- `icon_x_min = 55`, `icon_x_max = 175`: limits detection to the left icon column.
- `content_top = 245`: skips the iOS status bar and Octopus header.
- `content_bottom = height - 95`: avoids the home indicator area.
- `row_bbox.x = 40`, `row_bbox.width = 1060`: covers the visible transaction row area.
- `text_x = 205`: left edge of payee/date text.
- `payee_width = 650`: wide enough for long payees like `PAPER AND COFFEE LIMITED`, but still stops before the amount column.
- `date_width = 430`: date/time strings are shorter and should not include right-side amount text.
- `amount_bbox.x = 850`, `amount_bbox.width = 215`: isolates the right-aligned amount.

If the Octopus app changes spacing, these are the first values to revisit.

## Row Boundaries

Row boundaries are based on neighboring icon centers:

- Middle rows use the midpoint between previous/current/next icon centers.
- First and last rows fall back to an icon-relative height so clipped edge rows do not absorb unrelated content.
- Rows shorter than about `170px` at the reference scale are skipped as likely clipped/incomplete.

This is why multiline rows work: a taller gap between icon centers creates a taller row, rather than forcing every row into the old fixed `~200px` shape.

## Multiline Payees

The first version assumed one payee line and one date line. That failed for payees like:

```text
PAPER AND
COFFEE LIMITED
2026-05-28 13:44
```

The current logic detects dark text-line bands inside the row's middle text column:

- The last detected text line is treated as the date/time.
- All text lines above it are treated as the payee.

This keeps the pipeline UI-specific while handling one-line and two-line payees without hardcoding merchant names.

## Amount Sign and Decimal Recovery

Amounts are colored in the app:

- Red: outflow
- Green: inflow/top-up

OCR can drop punctuation or signs, for example reading `-2.0` as `20`. Because Octopus amounts display one decimal place, amount parsing inserts a decimal before the final digit when OCR returns digits without a decimal:

- `20` -> `2.0`
- `820` -> `82.0`
- `1000` -> `100.0`

The final sign comes from the amount color when available. This is safer than trusting OCR for `+`/`-`.

## Debug Artifacts

Annotated screenshots and field crops are part of the pipeline on purpose. They let us see whether an OCR failure came from:

- bad row detection,
- bad payee/date splitting,
- a too-small amount crop,
- or OCR misreading a good crop.

When the app UI changes, inspect `out/debug/*_annotated.png` before changing OCR settings.

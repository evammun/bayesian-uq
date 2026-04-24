# Dashboard Style Guide — "Fountain Pen in a Lab Coat"

## Core Vision

The New Yorker meets academic journal. Savile Row tailoring meets research — expensive simplicity that whispers rather than shouts.

**Quick test:** Would this look at home in both Nature journal and a Chelsea gallery opening?

## Typography

- **Headlines:** Playfair Display (serif) — gravitas without stuffiness
- **Body:** Inter (sans-serif) — readable, modern, not tech-bro
- **Weights:** Subtle — 300/400/500, never bold unless absolutely necessary
- **Spacing:** Generous line height and letter spacing — let text breathe

## Colour Palette

### Base

| Token | Hex | Use |
|-------|-----|-----|
| BG | `#FDFCFB` | Off-white background — like expensive paper, never pure white |
| TEXT | `#2C3E50` | Sophisticated charcoal — easier than black, more refined |
| LABEL | `#6B7280` | Warm gray for secondary text |
| GRAY | `#8B95A1` | Warm gray for muted elements |
| GRID | `#E8E4E0` | Chart gridlines, borders |

### Jewel Tones (charts and data visualization)

| Token | Hex | Use |
|-------|-----|-----|
| TEAL | `#2A8C8F` | Primary accent |
| DEEP_BLUE | `#4B7C92` | Secondary accent |
| SLATE | `#5B5E8D` | Tertiary |
| ROSE | `#CA4A7A` | Contrast/highlight |
| GOLD | `#D4A017` | Warm accent |
| PURPLE | `#6C4F7F` | Deep accent |
| SOFT_TEAL | `#65B2B5` | Light variant |

### Model Assignments

| Model | Colour |
|-------|--------|
| Qwen 3 | TEAL `#2A8C8F` |
| Gemma 4 | ROSE `#CA4A7A` |
| Qwen 3.5 | GOLD `#D4A017` |

### Method Assignments (adaptive sampling)

| Method | Colour |
|--------|--------|
| Product | TEAL `#2A8C8F` |
| Sum | ROSE `#CA4A7A` |
| Dirichlet MLE | GOLD `#D4A017` |
| MoM | SLATE `#5B5E8D` |
| MoM + Bayes | PURPLE `#6C4F7F` |

## Spatial Rules

- Whitespace is confidence — don't fill every corner
- Margins are generous but purposeful
- Dense where it matters, airy where it doesn't

## The "Never" List

- Never corporate PowerPoint vibes
- Never more than one decorative element per section
- Never sacrifice readability for style
- Never use effects that could be from a template
- Colour appears as a whisper, not a shout (except in charts where rich jewel tones are appropriate)

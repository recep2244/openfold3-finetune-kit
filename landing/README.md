# Landing page

A standalone, from-scratch marketing/landing page for **openfold3-finetune-kit** —
separate from the MkDocs documentation site. Research-instrument aesthetic
(pine-teal ink, phosphor-teal accent, amber CTA, Fraunces + IBM Plex), with an
animated molecular-network hero, scroll reveals, and a fully responsive,
keyboard-accessible, `prefers-reduced-motion`-aware layout.

```
landing/
├── index.html    # markup + content
├── styles.css    # design system + components
└── main.js       # hero canvas animation + scroll reveal (progressive enhancement)
```

## Preview locally
No build step — just open it, or serve the folder:

```bash
python -m http.server -d landing 8000   # then visit http://localhost:8000
```

## Deploy options
- **Project landing + docs subsite:** keep GitHub Pages on the MkDocs site and host this
  page elsewhere (e.g. the repo's `gh-pages` root, Netlify, or your portfolio).
- **As the Pages root:** publish `landing/` instead of the docs and move the MkDocs site to a
  `/docs/` subpath. (Ask before switching — the current Pages deploy serves the MkDocs site.)

The page is self-contained: fonts load from Google Fonts, and the social image is referenced
from the repo's `assets/social-preview.png`.

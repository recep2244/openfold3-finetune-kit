<div class="of3-hero" markdown="1">

# openfold3-finetune-kit

<p class="of3-hero__tagline">Target-specific fine-tuning for OpenFold3 — data preparation, training, and rigorous evaluation in one reproducible pipeline.</p>
<p class="of3-hero__by">Curated by <a href="https://recep2244.github.io/portfolio/">Recep Adiyaman</a></p>

<div class="of3-hero__cta" markdown="1">
[Get started](getting-started.md){ .md-button .md-button--primary } [View on GitHub](https://github.com/recep2244/openfold3-finetune-kit){ .md-button }
</div>

<ul class="of3-hero__chips">
<li>Apache-2.0</li>
<li>ColabFold MSAs · no databases</li>
<li>12 GB → 80 GB GPUs</li>
<li>Verified end-to-end</li>
</ul>

</div>

OpenFold3 is the Apache-2.0, all-atom co-folding model (an open reproduction of AlphaFold3). Like all co-folding models it interpolates well but extrapolates poorly: accuracy drops on chemistry far from its training distribution. This kit closes that gap for a *specific* target — a short, gentle fine-tune on ~10 protein–ligand complexes corrects systematic interface errors, then proves the gain on a held-out set.

```mermaid
flowchart LR
    A[PDB IDs<br/>train + held-out] --> B[Data prep<br/>structures + ColabFold MSAs]
    B --> C[Preflight check]
    C --> D[Fine-tune<br/>from of3-p2-155k]
    D --> E[Evaluate<br/>baseline vs tuned]
    E --> F[lDDT · DockQ · lDDT-PLI · ligand RMSD]
```

## Where to go next

- **[Getting started](getting-started.md)** — install OpenFold3 and verify the setup.
- **[Tutorial](tutorial.md)** — a full worked run on the PDE10A example target.
- **[How it works](pipeline.md)** — the four pipeline stages and the MSA/templates policy.
- **[Configuration](configuration.md)** — the fine-tune YAMLs and the knobs that matter.
- **[Usage reference](usage.md)** — every script and its environment variables.
- **[Troubleshooting](troubleshooting.md)** — the failure modes you are most likely to hit.

## Design principles

- **No alignment databases.** MSAs come from the ColabFold server; templates are off everywhere; training never touches the network.
- **Verify before you spend GPU hours.** A preflight check guards the known data pitfalls, and `verify_setup.sh` runs a real prediction end-to-end.
- **Nudge, don't retrain.** Low learning rate, short warmup, faster EMA, interface-biased crops — adapt to the target without catastrophic forgetting.

---

!!! quote "Why I built this"
    Target-specific OpenFold3 fine-tuning involves a lot of fiddly plumbing — preprocessing,
    MSAs, checkpoint-loading quirks, evaluation — and I wanted a path that just works and is
    reproducible end to end. Every config key here was checked against the OpenFold3 source and
    verified on real hardware. If it saves you an afternoon of debugging, it did its job.
    — *Recep Adiyaman*

## Author & related

Curated and maintained by **[Recep Adiyaman](https://recep2244.github.io/portfolio/)**.

- :material-web: [Portfolio](https://recep2244.github.io/portfolio/) — about me and other work
- :fontawesome-brands-github: [GitHub profile](https://github.com/recep2244) · [all repositories](https://github.com/recep2244?tab=repositories)
- :fontawesome-solid-dna: [OpenFold3 (upstream)](https://github.com/aqlaboratory/openfold-3) — the model this kit fine-tunes
- :material-source-repository: [Project repository](https://github.com/recep2244/openfold3-finetune-kit)

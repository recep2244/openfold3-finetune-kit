# Configuration

The kit ships four fine-tune configs in `configs/`. `run_all.sh` selects one via
`GPU_PROFILE` and fills in your data paths automatically; you rarely edit them by hand.

| Config | Target hardware | Notes |
|---|---|---|
| `finetune_lowN_single_gpu.yml` | One 80 GB GPU | The reference low-N recipe |
| `finetune_single_target.yml` | One 80 GB GPU | Adds an anti-forgetting replay set |
| `finetune_multi_gpu.yml` | Multiple GPUs | Scale past a few dozen structures |
| `finetune_test_12gb.yml` | 12–16 GB GPU | Tiny smoke test; not a real fine-tune |

## The low-N recipe — and why

These overrides turn full training into a gentle *nudge* (defaults shown for contrast):

| Setting | Value | Default | Rationale |
|---|---|---|---|
| `optimizer.learning_rate` | `3e-4` | `1.8e-3` | 6× lower — adapt without overwriting prior knowledge |
| `lr_scheduler.warmup_no_steps` | `50` | `1000` | A 1000-step warmup never finishes in a ~350-step run |
| `ema.decay` | `0.99` | `0.999` | Let the averaged weights actually move |
| `template.n_templates` | `0` | — | Templates off (matches the published recipe) |
| `crop.token_crop.token_budget` | `384` | `640` | Lower budget fits smaller GPUs |
| `crop.crop_weights.spatial_interface` | `0.4` | — | Bias crops toward the binding interface |
| `loss.loss_weights.bond` | `4.0` | — | Improves ligand geometry |

## Checkpoint loading

The released `of3-p2-155k.pt` is a bare state dict. The loader auto-detects this and loads
it directly — no `.pt` → `.ckpt` conversion:

```yaml
ckpt_load_settings:
  manual_checkpoint_loading: true   # required to load a bare .pt
  init_from_ema_weights: false      # the .pt has no EMA block
  restore_lr_scheduler: false       # fresh schedule for the fine-tune
  restore_time_step: false          # start global_step at 0
  strict_loading: false             # tolerate head/key differences
```

## Data paths

`run_all.sh` rewrites the placeholder paths (`/shared/openfold3/...`) in the chosen config
to point at the artifacts produced under your `WORK` directory, so the configs are portable
as-is. The full key-by-key rationale, with upstream source references, is in
[Background & rationale](background.md).

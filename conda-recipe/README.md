# conda packaging

Two recipes live here, intentionally redundant:

| File          | Used by                                  |
| ------------- | ---------------------------------------- |
| `meta.yaml`   | conda-forge (`staged-recipes` PR #33314) |
| `recipe.yaml` | personal channel build via `rattler-build` (`.github/workflows/conda.yml`) |

Both build the same package; the existing conda-forge submission keeps the older `meta.yaml` to avoid churn on that PR.

## Personal anaconda.org channel

Built and uploaded by `.github/workflows/conda.yml` to <https://anaconda.org/isaaccorley/gdal-conv2d> on every `v*` tag push (or manually via the `workflow_dispatch` trigger with `upload: true`).

End users install with:

```bash
conda install -c isaaccorley -c conda-forge gdal-conv2d
```

`-c conda-forge` is needed because `libgdal` is a runtime dependency.

### One-time setup

1. Generate an [anaconda.org API token](https://anaconda.org/isaaccorley/settings/access) with scopes `api:read` and `api:write` and the package matching `gdal-conv2d` (or `*` to keep it simple).
2. Add it to the repo as a secret named `ANACONDA_TOKEN`:

   ```bash
   gh secret set ANACONDA_TOKEN -R isaaccorley/gdal-unet
   ```

3. Push a `v*` tag, or run the workflow manually with `upload: true`.

### Local dry-run

Build only, no upload:

```bash
pixi global install rattler-build
rattler-build build --recipe conda-recipe/recipe.yaml --channel conda-forge
```

Built `.conda` files land in `output/`.

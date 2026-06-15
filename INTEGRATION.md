# Integrating the top-9 fast intranode combine (AccumNum=9)

This branch adds the `_vec8_top9` (AccumNum=9) fast fp8-blockwise weightless
intranode combine kernels so that the fast path is taken under shared-expert
fusion (topk=9: 8 routed + 1 fused shared expert), instead of falling back to
the slow generic weightless path.

- **Branch:** `rbrugaro/ep-combine-vec8-top9-shared-fusion-v1.2.0`
- **Commit:** `ba35972` (parent `bf99bdf` = mori v1.2.0)
- **Diff:** 6 files, math-identical to the generic weightless path (no accuracy change)
- **Base image it targets:** `rocm/sgl-dev:sglang-0.5.12.post1-rocm720-mi35x-mori-0610-moe`
  (ships mori at exactly `bf99bdf`, so integrating this branch is a clean +1-commit fast-forward)

The patch only activates under: shared-expert fusion (`numExpertPerToken == 9`),
fp8-blockwise quant, weightless combine, `hiddenDim % 512 == 0`, `worldSize > 4`,
and fp8 scale block size in {128, 256}. Outside that it is a no-op, so it is safe
to bake in unconditionally.

## Step 1 — update the mori source in the image to this commit

```bash
# inside a container started from sglang-0.5.12.post1-rocm720-mi35x-mori-0610-moe
cd /sgl-workspace/mori
git remote add rbrugaro https://github.com/rbrugaro-amd/mori.git
git fetch rbrugaro rbrugaro/ep-combine-vec8-top9-shared-fusion-v1.2.0
git checkout rbrugaro/ep-combine-vec8-top9-shared-fusion-v1.2.0   # fast-forward from bf99bdf, +1 commit
git rev-parse --short HEAD    # expect ba35972
```

No submodule re-init is needed: this commit does not touch submodules, and the
image already has them populated.

## Step 2 — rebuild the C++ extension (host gate in launch.cpp)

```bash
cd /sgl-workspace/mori
MORI_GPU_ARCHS=gfx950 pip install --no-build-isolation --force-reinstall --no-deps .
```

## Step 3 — refresh the JIT device kernels (ep_intranode.hsaco)

The JIT cache is keyed by a sha256 of the source, so it auto-recompiles on next
use. To force it eagerly and verify the new symbols are present:

```bash
rm -rf /root/.mori/jit
python -c "from mori.jit.core import compile_genco; print(compile_genco('ep_intranode', source_dir='src/ops/kernels'))"
HS=$(find /root/.mori -name ep_intranode.hsaco | head -1)
strings "$HS" | grep -o 'noweight_block[0-9]*_vec8_top9' | sort -u
# expect:
#   noweight_block128_vec8_top9
#   noweight_block256_vec8_top9
```

## Step 4 — bake into a new image

```bash
docker commit <container> rocm/sgl-dev:...-mori-0610-moe-accum9
```

Or add Steps 1–3 as a layer in the Dockerfile after the base image.

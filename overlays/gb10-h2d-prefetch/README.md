# GB10 attested-reader H2D prefetch

Overlaps attested `mt_fread` with trellis-arena `copy_` by prefetching a
small queue of CPU tensors (default `VLLM_EXL3_H2D_PREFETCH=4`). Same
per-tensor attestation, still one shard at a time.

Measured on the SAGE 3.30bpw EP2 pair: 40 mixed-K layers ~5 min vs ~32 min
serial get_tensor→H2D. Prefetch 16 matched 4 on cadence and died at the UMA
cliff; keep 4.

Apply inside the image or bind-mount the reader over site-packages, then run
`apply.sh` so `exl3_reader_lock.json` hashes this file. Pair with
`VLLM_EXL3_MEM_WATERFALL=0` (see `profiles/tp2-disk-engram.env`).

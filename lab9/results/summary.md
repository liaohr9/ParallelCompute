# Benchmark summary

- Overall best: tiled_pad n=2048 config=64x8 bandwidth=4097.985 GB/s time=0.008188 ms
- Best rows on n=2048:
  - tiled_pad config=64x8 bandwidth=4097.985 GB/s time=0.008188 ms
  - tiled_nopad config=16x4 bandwidth=3278.380 GB/s time=0.010235 ms
  - naive_read_strided config=16x16 bandwidth=3278.379 GB/s time=0.010235 ms
  - naive_write_strided config=16x8 bandwidth=1865.881 GB/s time=0.017983 ms
- Large-size tiled_pad best rows:
  - n=4096 config=64x8 bandwidth=1826.176 GB/s time=0.073497 ms
  - n=8192 config=64x16 bandwidth=1513.064 GB/s time=0.354824 ms
  - n=12288 config=64x16 bandwidth=1500.024 GB/s time=0.805294 ms
  - n=16384 config=64x16 bandwidth=1503.486 GB/s time=1.428336 ms
  - n=24576 config=64x16 bandwidth=1493.328 GB/s time=3.235617 ms
  - n=32768 config=64x16 bandwidth=1500.650 GB/s time=5.724144 ms

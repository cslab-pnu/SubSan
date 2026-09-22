# SubSan

SubSan: Efficient Memory Error Detection via Subtraction Checks.

This is the artifact repository of our paper will be presented at the ACM SIGOPS Annual Technical Conference 2026 (ATC '26).

## Tested Environments

### Arm TBI (Top Byte Ignore, 8-bit tag)
- Machine: MacBook Pro (Apple M1), 32 GB RAM
- OS: Fedora Asahi Remix 42
- Kernel: Linux 6.8
- libc: glibc 2.39
- Architecture: AArch64 (Apple Silicon)
- Pointer tag: **8 bits** (bits 63:56)

### Intel LAM (Linear Address Masking, 6-bit tag)
- Machine: Intel Core Ultra 9 285K (Arrow Lake), 64 GB RAM
- OS: Ubuntu 24.04 LTS
- Kernel: Linux ≥ 6.4 (`CONFIG_ADDRESS_MASKING=y`)
- libc: glibc 2.39
- Architecture: x86_64 (LAM_U57 capable: Sapphire Rapids / Arrow Lake)
- Pointer tag: **6 bits** (LAM_U57: bits 62:57; bit 63 reserved must be 0)

## Prerequisites

```
# Fedora / Fedora Asahi Remix
sudo dnf install -y gcc gcc-c++ make cmake ninja-build python3 git zlib-devel libzstd-devel libedit-devel ncurses-devel

# Ubuntu / Debian
sudo apt install -y build-essential cmake ninja-build python3 git zlib1g-dev libzstd-dev libedit-dev libncurses-dev
```

- Requires CMake ≥ 3.20, Python ≥ 3.8, GCC ≥ 7.1.
- Building LLVM needs ~60 GB of free disk and ≥ 16 GB of RAM (with less RAM, add `-DLLVM_PARALLEL_LINK_JOBS=1`).

## How to install

```
# Arm TBI
./install-llvm-tbi.sh

# Intel LAM
./install-llvm-lam.sh
```

## Running Examples (SubSan)
```
cd examples
./test.sh
```

Expected Output:

```
=== subsan_shadow test ===
Compiler : .../llvm-build/bin/clang
Arch     : aarch64
CFLAGS   : -O2 -g -no-pie -fsanitize=safe-stack ...
LDFLAGS  : -fsanitize=safe-stack -fuse-ld=lld -no-pie -z muldefs ...

Compiling test programs...
Done.

--- [buffer-overflow]  valid index (expected: normal exit) ---
obj: 0x52000e0000000108
obj[40] access...
0
Exit: 0

--- [buffer-overflow]  invalid index (expected: subsan trap) ---
obj: 0x98000e0000000108
obj[64] access...
./test.sh: line 85: 2004503 Trace/breakpoint trap   "$@"
Exit: 133

--- [buffer-underflow] (expected: subsan trap) ---
obj: 0x64000e0000000108
obj[-4] access...
./test.sh: line 85: 2004507 Trace/breakpoint trap   "$@"
Exit: 133

--- [use-after-free] (expected: subsan trap) ---
obj: 0xe6000e0000000108
obj 0xe6000e0000000108 deallocated...
use-after-free!!!
./test.sh: line 85: 2004518 Trace/breakpoint trap   "$@"
Exit: 133
```

## SubSan Benchmark Results

> The SPEC results below are measured on the **Arm TBI** environment (Apple M1 / Fedora Asahi Remix).

### MSET


| Class    | Bug type            | Total | Detected | Undetected | Detection rate |
|----------|---------------------|------:|---------:|-----------:|---------------:|
| Temporal | Misuse-of-free      |    20 |       20 |          0 | 100.00% (20/20) |
| Temporal | Double-free         |     4 |        4 |          0 | 100.00% (4/4)   |
| Temporal | Use-after-*         |    16 |       16 |          0 | 100.00% (16/16) |
| **Temporal** | **Total**      | **40**| **40**   | **0**      | **100.00% (40/40)** |
| Spatial  | Linear OOBA         |    90 |       72 |         18 | 80.00% (72/90)  |
| Spatial  | Non-Linear OOBA     |    72 |       54 |         18 | 75.00% (54/72)  |
| Spatial  | Type Confusion OOBA |    30 |       24 |          6 | 80.00% (24/30)  |
| **Spatial** | **Total**       | **192**| **150**  | **42**     | **78.12% (150/192)** |
| **Overall** | **Total**      | **232**| **190**  | **42**     | **81.90% (190/232)** |

### SPEC CPU2006

|       | **Runtime** |        |             | **MaxRSS** |         |             |
|---------------|:-------:|:------:|:-----------:|:-------:|:-------:|:-----------:|
| **Benchmark** | **Base** | **SubSan** | **SubSan/Base** | **Base** | **SubSan** | **SubSan/Base** |
| 400.perlbench | 156     | 321    | 2.06        | 679504  | 1070048 | 1.57        |
| 401.bzip2     | 262     | 408    | 1.56        | 870208  | 1307072 | 1.50        |
| 403.gcc       | 102     | 336    | 3.29        | 907536  | 1488156 | 1.64        |
| 429.mcf       | 144     | 218    | 1.51        | 1716816 | 2574896 | 1.50        |
| 433.milc      | 105     | 156    | 1.49        | 697040  | 1056288 | 1.52        |
| 444.namd      | 154     | 220    | 1.43        | 49184   | 50752   | 1.03        |
| 445.gobmk     | 202     | 305    | 1.51        | 29808   | 45232   | 1.52        |
| 447.dealII    | 121     | 195    | 1.61        | 815216  | 1248416 | 1.53        |
| 450.soplex    | 92.7    | 153    | 1.65        | 432448  | 1138800 | 2.63        |
| 453.povray    | 51.5    | 113    | 2.19        | 7280    | 10400   | 1.43        |
| 456.hmmer     | 165     | 182    | 1.10        | 26160   | 53712   | 2.05        |
| 458.sjeng     | 262     | 501    | 1.91        | 180264  | 277308  | 1.54        |
| 462.libquantum| 87.2    | 108    | 1.24        | 99360   | 203552  | 2.05        |
| 464.h264ref   | 210     | 367    | 1.75        | 66876   | 127844  | 1.91        |
| 470.lbm       | 85.8    | 97.8   | 1.14        | 419904  | 629296  | 1.50        |
| 471.omnetpp   | 161     | 287    | 1.78        | 175776  | 261776  | 1.49        |
| 473.astar     | 199     | 291    | 1.46        | 334688  | 502496  | 1.50        |
| 482.sphinx3   | 198     | 281    | 1.42        | 43920   | 71184   | 1.62        |
| 483.xalancbmk | 105     | 254    | 2.42        | 428480  | 647680  | 1.51        |
| **geomean**   | 138.87  | 229.46 | **1.652**   | 195849  | 314458  | **1.606**   |



### SPEC CPU2017

|           | **Runtime** |        |             | **MaxRSS** |         |             |
|-------------------|:-------:|:------:|:-----------:|:--------:|:-------:|:-----------:|
| **Benchmark** | **Base** | **SubSan** | **SubSan/Base** | **Base** | **SubSan** | **SubSan/Base** |
| 600.perlbench_s   | 228     | 381    | 1.67        | 206736   | 304128   | 1.47        |
| 602.gcc_s         | 260     | 511    | 1.97        | 7836624  | 12007008 | 1.53        |
| 605.mcf_s         | 373     | 561    | 1.50        | 4055584  | 6832304  | 1.68        |
| 619.lbm_s         | 432     | 496    | 1.15        | 3300896  | 4951088  | 1.50        |
| 620.omnetpp_s     | 279     | 516    | 1.85        | 246320   | 368736   | 1.50        |
| 623.xalancbmk_s   | 200     | 359    | 1.80        | 490288   | 739440   | 1.51        |
| 625.x264_s        | 96.3    | 153    | 1.59        | 163936   | 252512   | 1.54        |
| 631.deepsjeng_s   | 218     | 311    | 1.43        | 7046240  | 10568256 | 1.50        |
| 638.imagick_s     | 3693    | 4288   | 1.16        | 7368304  | 10761520 | 1.46        |
| 641.leela_s       | 275     | 465    | 1.69        | 28096    | 39184    | 1.39        |
| 644.nab_s         | 988     | 1469   | 1.49        | 570704   | 867264   | 1.52        |
| 657.xz_s          | 2177    | 2920   | 1.34        | 15643792 | 23564784 | 1.51        |
| **geomean**       | 412.74  | 632.30 | **1.532**   | 1154245  | 1740825  | **1.508**   |

## License

SubSan is released under the Apache License 2.0 with LLVM Exceptions,
the same license as the LLVM Project, on which it is built.

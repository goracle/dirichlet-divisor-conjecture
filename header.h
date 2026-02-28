#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <atomic>
#include <cstring>
#include <omp.h>
#include <gmp.h>
#include <hdf5.h>

#include <quadmath.h>




static const __float128 PI_HALF_VAL = 1.57079632679489661923132169163975144Q; // π/2

typedef __uint128_t u128;
typedef unsigned long long ull;
using u128 = unsigned __int128;
using i128 = __int128;





void run_asymptotic_table(u128 cutoff, u128 max_n, double ratio,
                          const char *hdf5_path);


i128 a_hoying_u128(u128 n);

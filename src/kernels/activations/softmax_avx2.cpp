#include <cmath>
#include <cstddef>
#include <algorithm>
#include <immintrin.h>

namespace activations {

void softmax_row_avx2(const float* input, float* output, int n) {
    if (n <= 0) return;
    // Numerically stable scalar path: find max, subtract, exp, normalize.
    float maxv = input[0];
    for (int i = 1; i < n; ++i) if (input[i] > maxv) maxv = input[i];

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(input[i] - maxv));
        output[i] = static_cast<float>(e);
        sum += e;
    }
    if (sum == 0.0) {
        // Prevent division by zero; distribute uniformly
        float v = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) output[i] = v;
        return;
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < n; ++i) output[i] = static_cast<float>(output[i] * inv);
}

} // namespace activations

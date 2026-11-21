# Analytical-coordinate interpolation

`LookupPolicy(interpolation_space="analytical")` blends measured latency anchors in the native analytical model's growth coordinate.
For an axis value between two measured anchors, the interpolation weight is `(A(target) - A(lower)) / (A(upper) - A(lower))`, where `A` is the supplied analytical scaler.
The measured latency uses that weight between its lower and upper anchors.
An alpha-beta communication model therefore interpolates linearly in bytes between latency anchors, preserving a fixed launch term and a bandwidth term.
The existing default `interpolation_space="axis"` retains its declared-axis behavior, including logarithmic-axis weighting.

The analytical option applies only when a scaler supplies finite, monotonic coordinates that bracket the target.
Otherwise lookup falls back to declared-axis interpolation and exports the `axis_interpolation_fallback` flag.
Successful analytical weighting exports `analytical_interpolated` alongside the ordinary interpolation flags.
Exact lookup and out-of-range analytical extrapolation retain their existing behavior.

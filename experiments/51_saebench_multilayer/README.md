# Experiment 51: SAEBench sparse probing at L9/L18/L27

Five SAE architectures (standard, adaptive cosine, per-feature cosine, NoC, and an independently trained reference) are evaluated on the SAEBench sparse probing benchmark across three transformer depths (L9, L18, L27) on Qwen3-8B. The independent reference SAE (separately trained BatchTopK, 500M tokens, different codebase) validates that the standard baseline is not implementation-specific.

## Results

Sparse probing top-1 accuracy:

| SAE | L9 (50M) | L18 (500M) | L27 (50M) | Mean |
|-----|----------|------------|-----------|------|
| standard | 0.666 | 0.678 | 0.765 | 0.703 |
| adaptive_l2 | 0.782 | 0.800 | 0.849 | 0.810 |
| perfeature_l2 | 0.786 | 0.815 | 0.841 | 0.814 |
| no_C | 0.769 | 0.783 | 0.802 | 0.785 |
| independent_ref | 0.613 | 0.679 | 0.762 | 0.685 |

Cosine advantage over standard (top-1 percentage points):

| | L9 | L18 | L27 |
|---|---|---|---|
| adaptive_l2 | +11.6 | +12.2 | +8.4 |
| perfeature_l2 | +12.0 | +13.7 | +7.6 |
| no_C | +10.3 | +10.5 | +3.7 |

Full sparse probing metrics:

| SAE | Layer | top-1 | top-2 | top-5 | test_acc |
|-----|-------|-------|-------|-------|----------|
| standard | 9 | 0.666 | 0.708 | 0.796 | 0.891 |
| adaptive_l2 | 9 | 0.782 | 0.822 | 0.863 | 0.950 |
| perfeature_l2 | 9 | 0.786 | 0.813 | 0.877 | 0.952 |
| no_C | 9 | 0.769 | 0.781 | 0.814 | 0.925 |
| independent_ref | 9 | 0.613 | 0.666 | 0.773 | 0.952 |
| standard | 18 | 0.678 | 0.728 | 0.789 | 0.944 |
| adaptive_l2 | 18 | 0.800 | 0.853 | 0.891 | 0.957 |
| perfeature_l2 | 18 | 0.815 | 0.852 | 0.883 | 0.959 |
| no_C | 18 | 0.783 | 0.798 | 0.805 | 0.913 |
| independent_ref | 18 | 0.679 | 0.742 | 0.780 | 0.945 |
| standard | 27 | 0.765 | 0.836 | 0.892 | 0.956 |
| adaptive_l2 | 27 | 0.849 | 0.886 | 0.910 | 0.957 |
| perfeature_l2 | 27 | 0.841 | 0.877 | 0.898 | 0.955 |
| no_C | 27 | 0.802 | 0.846 | 0.868 | 0.932 |
| independent_ref | 27 | 0.762 | 0.818 | 0.875 | 0.956 |

Core eval metrics:

| SAE | KL score | CE score | FVE | cossim | L0 | alive% |
|-----|----------|----------|-----|--------|-----|--------|
| adaptive_l2 L18 | 0.985 | 0.993 | 0.552 | 0.925 | 85.6 | 99.2% |
| perfeature_l2 L18 | 0.985 | 0.993 | 0.510 | 0.925 | 84.5 | 99.1% |
| no_C L18 | 0.984 | 0.991 | 0.223 | 0.923 | 85.2 | 92.0% |
| independent_ref L18 | 0.988 | 0.991 | -91.5 | 0.926 | 84.7 | 99.1% |
| standard L27 | 0.977 | 0.986 | 0.853 | 0.934 | 80.2 | 99.7% |
| adaptive_l2 L27 | 0.977 | 0.984 | 0.865 | 0.935 | 78.9 | 98.0% |
| perfeature_l2 L27 | 0.967 | 0.977 | 0.815 | 0.921 | 77.4 | 16.6% |
| no_C L27 | 0.977 | 0.977 | 0.852 | 0.932 | 79.8 | 89.4% |
| independent_ref L27 | 0.985 | 0.984 | 0.430 | 0.945 | 83.8 | 98.8% |

Per-dataset top-1 accuracy, Layer 9:

| Dataset | standard | adaptive | perfeature | no_C | independent_ref |
|---------|----------|----------|------------|------|-----------------|
| bias_in_bios set1 | 0.721 | 0.875 | 0.872 | 0.812 | 0.612 |
| bias_in_bios set2 | 0.674 | 0.851 | 0.867 | 0.837 | 0.548 |
| bias_in_bios set3 | 0.742 | 0.777 | 0.773 | 0.754 | 0.679 |
| amazon_reviews | 0.754 | 0.758 | 0.743 | 0.690 | 0.737 |
| amazon_sentiment | 0.599 | 0.593 | 0.596 | 0.592 | 0.594 |
| github_code | 0.538 | 0.649 | 0.697 | 0.739 | 0.539 |
| ag_news | 0.615 | 0.804 | 0.782 | 0.730 | 0.556 |
| europarl | 0.684 | 0.953 | 0.960 | 0.996 | 0.638 |

Per-dataset top-1 accuracy, Layer 18:

| Dataset | standard | adaptive | perfeature | no_C | independent_ref |
|---------|----------|----------|------------|------|-----------------|
| bias_in_bios set1 | 0.669 | 0.798 | 0.862 | 0.786 | 0.644 |
| bias_in_bios set2 | 0.559 | 0.762 | 0.758 | 0.766 | 0.623 |
| bias_in_bios set3 | 0.612 | 0.778 | 0.766 | 0.763 | 0.667 |
| amazon_reviews | 0.712 | 0.713 | 0.715 | 0.675 | 0.713 |
| amazon_sentiment | 0.915 | 0.924 | 0.880 | 0.795 | 0.901 |
| github_code | 0.514 | 0.700 | 0.813 | 0.784 | 0.520 |
| ag_news | 0.704 | 0.736 | 0.735 | 0.720 | 0.705 |
| europarl | 0.735 | 0.986 | 0.992 | 0.978 | 0.658 |

Per-dataset top-1 accuracy, Layer 27:

| Dataset | standard | adaptive | perfeature | no_C | independent_ref |
|---------|----------|----------|------------|------|-----------------|
| bias_in_bios set1 | 0.840 | 0.863 | 0.883 | 0.847 | 0.864 |
| bias_in_bios set2 | 0.766 | 0.787 | 0.794 | 0.833 | 0.725 |
| bias_in_bios set3 | 0.752 | 0.806 | 0.823 | 0.711 | 0.749 |
| amazon_reviews | 0.773 | 0.788 | 0.797 | 0.739 | 0.732 |
| amazon_sentiment | 0.909 | 0.926 | 0.910 | 0.846 | 0.930 |
| github_code | 0.522 | 0.890 | 0.781 | 0.826 | 0.518 |
| ag_news | 0.742 | 0.777 | 0.743 | 0.728 | 0.741 |
| europarl | 0.821 | 0.954 | 0.998 | 0.888 | 0.835 |

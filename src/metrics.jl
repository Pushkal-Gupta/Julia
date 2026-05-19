"""
    Metrics

Quantitative comparison of edge maps. Edge localization is rarely
pixel-exact — smoothing, sub-pixel ambiguity, and noise all shift
detected edges by a pixel or two. A "tolerance" parameter handles
that: a predicted edge counts as correct if any ground-truth edge
lies within `tolerance` pixels of it.

Operationally that means dilating the ground truth by `tolerance` and
intersecting, which is exactly what the implementation here does.
"""
module Metrics

using ..Filters: binary_dilate

export edge_match_stats, iou_score

"""
    edge_match_stats(predicted::BitMatrix, gt::BitMatrix; tolerance=1)
        -> (precision, recall, f1)

Compare a predicted binary edge map against ground truth using
tolerance-based matching:

- **Precision** = fraction of predicted-edge pixels whose dilated
  ground-truth neighborhood contains a ground-truth pixel. ("Of the
  edges I called, what fraction were near a real edge?")
- **Recall** = fraction of ground-truth pixels whose dilated
  prediction neighborhood contains a predicted pixel. ("Of the real
  edges, what fraction did I find?")
- **F1** = harmonic mean.

`tolerance` is in pixels (Chebyshev distance, i.e. an `(2t+1) × (2t+1)`
box). `tolerance = 0` is exact matching.
"""
function edge_match_stats(predicted::BitMatrix, gt::BitMatrix;
                          tolerance::Integer = 1)
    size(predicted) == size(gt) || throw(DimensionMismatch(
        "predicted and gt must share size; got $(size(predicted)) and $(size(gt))"))
    n_pred = sum(predicted)
    n_gt = sum(gt)

    if n_pred == 0 && n_gt == 0
        return (1.0, 1.0, 1.0)        # both empty: trivially perfect
    elseif n_pred == 0
        return (0.0, 0.0, 0.0)        # called nothing → 0 precision and recall
    elseif n_gt == 0
        return (0.0, 0.0, 0.0)        # nothing to find → can't score

    end

    gt_dilated   = binary_dilate(gt;        radius = tolerance)
    pred_dilated = binary_dilate(predicted; radius = tolerance)

    precision = sum(predicted .& gt_dilated)   / n_pred
    recall    = sum(gt        .& pred_dilated) / n_gt
    f1 = (precision + recall) == 0 ? 0.0 :
         2 * precision * recall / (precision + recall)
    return (precision, recall, f1)
end

"""
    iou_score(a::BitMatrix, b::BitMatrix) -> Float64

Intersection-over-union (Jaccard index) of two binary masks. Useful as
a single-number summary when I don't care to separate precision from
recall. Empty masks both ways → 1.0; either-but-not-both empty → 0.0.
"""
function iou_score(a::BitMatrix, b::BitMatrix)
    size(a) == size(b) || throw(DimensionMismatch(
        "a and b must share size; got $(size(a)) and $(size(b))"))
    inter = sum(a .& b)
    uni   = sum(a .| b)
    uni == 0 && return 1.0
    return inter / uni
end

end # module Metrics

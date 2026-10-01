using Microsoft.ML.OnnxRuntime;

namespace LayaDotNet;

public record Inputs(
    int K,
    int L,
    int N,
    long[] InputIds,
    long[] Attention,
    long[] MarkerPos,
    bool[] MarkerMask,
    long[] QType,
    int NTokens,
    List<SequenceItem> Items,
    List<NamedOnnxValue> Tensors
);
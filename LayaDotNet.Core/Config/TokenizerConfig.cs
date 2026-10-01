using System.Text.Json.Serialization;

namespace LayaDotNet.Config;

public record TokenizerConfig
(
    [property: JsonPropertyName("cls_token")] string ClsToken,
    [property: JsonPropertyName("mask_token")] string MaskToken,
    [property: JsonPropertyName("pad_token")] string PadToken,
    [property: JsonPropertyName("sep_token")] string SepToken,
    [property: JsonPropertyName("unk_token")] string UnkToken
);

using System.Text.Json.Serialization;

namespace LayaDotNet.Config;

public record LayaConfig
(
    [property: JsonPropertyName("max_len")] int MaxLength,
    [property: JsonPropertyName("head_max_len")] int HeadMaxLength,
    [property: JsonPropertyName("temperature")] float[] Temperature,
    [property: JsonPropertyName("temperature_by_options")] Dictionary<string, float> TemperatureByOptions
);

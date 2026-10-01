namespace LayaDotNet;

public class LayaOptions
{
    public string DownloadPath { get; set; } = "./onnx-laya";
    public string HfRepo { get; set; } = "receptron/laya-onnx";
    public bool SkipHfDownload { get; set; }
    public string TokenizerPath { get; set; } = "./onnx-laya/tokenizer/tokenizer.json";
    public string TokenizerConfigPath { get; set; } = "./onnx-laya/tokenizer/tokenizer_config.json";
    public string LayaConfigPath { get; set; } = "./onnx-laya/laya_config.json";

    public static LayaOptions Build(string hfRepo, string downloadPath)
    {
        return new LayaOptions
        {
            DownloadPath = downloadPath,
            HfRepo = hfRepo,
            SkipHfDownload = false,
            TokenizerPath = $"{downloadPath}/tokenizer/tokenizer.json",
            TokenizerConfigPath = $"{downloadPath}/tokenizer/tokenizer_config.json",
            LayaConfigPath = $"{downloadPath}/laya_config.json"
        };
    }
}

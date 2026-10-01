using LayaDotNet.Config;
using LayaDotNet.Questions;
using Tokenizers.HuggingFace.Tokenizer;
using Microsoft.ML.OnnxRuntime;

namespace LayaDotNet.Cpu;

public class Laya(InferenceSession session, Tokenizer tokenizer, LayaConfig layaConfig, SpecialIds specialIds) : LayaBase(tokenizer, layaConfig, specialIds), IDisposable
{
    public static async Task<Laya> LoadAsync(LayaOptions? options = null)
    {
        var layaBase = await LoadBaseAsync(options);
        var sessionOptions = new SessionOptions();
        var layaSession = new InferenceSession($"{layaBase.LayaOptions.DownloadPath}/laya.onnx", sessionOptions);
        return new(layaSession, layaBase.Tokenizer, layaBase.LayaConfig, layaBase.SpecialIds);
    }

    public SystemOneResponse Predict(string state, List<QuestionBase> questions)
    {
        var inputData = PrepareInputTensors(state, questions, Tokenizer, SpecialIds, LayaConfig);
        using var outs = session.Run(inputData.Tensors);
        var logits = outs.First(x => x.Name == "logits").AsTensor<float>();
        var actProbs = outs.First(x => x.Name == "act_probs").AsTensor<float>();
        var nAct = session.OutputMetadata["act_probs"].Dimensions[1];

        return ProcessModelOutput(inputData, LayaConfig, nAct, logits, actProbs);
    }

    public void Dispose()
    {
        session.Dispose();
        Tokenizer.Dispose();
    }
}

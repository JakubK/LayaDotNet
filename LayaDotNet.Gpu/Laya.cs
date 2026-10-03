using LayaDotNet.Config;
using LayaDotNet.Questions;
using Tokenizers.HuggingFace.Tokenizer;
using Microsoft.ML.OnnxRuntime;

namespace LayaDotNet.Gpu;

public class Laya(InferenceSession session, Tokenizer tokenizer, LayaConfig layaConfig, SpecialIds specialIds) : IDisposable
{
    public static async Task<Laya> LoadAsync(LayaOptions? options = null)
    {
        var layaBase = await LayaBase.LoadBaseAsync(options);
        var sessionOptions = new SessionOptions();
        sessionOptions.AppendExecutionProvider_CUDA();
        var layaSession = new InferenceSession($"{layaBase.Options.DownloadPath}/laya.onnx", sessionOptions);
        return new(layaSession, layaBase.Tokenizer, layaBase.Config, layaBase.SpecialIds);
    }


    public SystemOneResponse Predict(dynamic state, List<QuestionBase> questions) => Predict(Utils.SerializeState(state), questions);
    public SystemOneResponse Predict(string state, List<QuestionBase> questions)
    {
        var inputData = LayaBase.PrepareInputTensors(state, questions, tokenizer, specialIds, layaConfig);
        using var outs = session.Run(inputData.Tensors);
        var logits = outs.First(x => x.Name == "logits").AsTensor<float>();
        var actProbs = outs.First(x => x.Name == "act_probs").AsTensor<float>();
        var nAct = session.OutputMetadata["act_probs"].Dimensions[1];

        return LayaBase.ProcessModelOutput(inputData, layaConfig, nAct, logits, actProbs);
    }

    public void Dispose()
    {
        session.Dispose();
        tokenizer.Dispose();
    }
}

using System.Text.Json;
using CliWrap;
using LayaDotNet.Answers;
using LayaDotNet.Config;
using LayaDotNet.Questions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.HuggingFace.Tokenizer;

namespace LayaDotNet;

public class Laya(InferenceSession session, Tokenizer tokenizer, LayaConfig layaConfig, SpecialIds specialIds) : IDisposable
{
    public static async Task<Laya> LoadAsync(LayaOptions? options = null)
    {
        if (options == null)
        {
            options = LayaOptions.Build("receptron/laya-onnx", "./laya-onnx");
        }

        if (!options.SkipHfDownload)
        {
            await Cli.Wrap("hf")
                .WithArguments(["download", options.HfRepo, "--local-dir", options.DownloadPath])
                .WithStandardErrorPipe(PipeTarget.ToDelegate(System.Console.Error.WriteLine))
                .WithStandardOutputPipe(PipeTarget.ToDelegate(System.Console.WriteLine))
                .ExecuteAsync();
        }
        
        var layaSession = new InferenceSession($"{options.DownloadPath}/laya.onnx");
        var layaTokenizer = Tokenizer.FromFile(options.TokenizerPath);

        var layaConfigText = await File.ReadAllTextAsync(options.LayaConfigPath);
        var layaConfig = JsonSerializer.Deserialize<LayaConfig>(layaConfigText)!;

        var tokenizerConfigText = await File.ReadAllTextAsync(options.TokenizerConfigPath);
        var tokenizerConfig = JsonSerializer.Deserialize<TokenizerConfig>(tokenizerConfigText)!;

        var specialIds = new SpecialIds(
            TokenToId(layaTokenizer, tokenizerConfig.ClsToken),
            TokenToId(layaTokenizer, tokenizerConfig.SepToken),
            TokenToId(layaTokenizer, tokenizerConfig.MaskToken),
            TokenToId(layaTokenizer, tokenizerConfig.PadToken),
            tokenizerConfig.MaskToken
        );

        return new(layaSession, layaTokenizer, layaConfig, specialIds);
    }

    public SystemOneResponse Predict(string state, List<QuestionBase> questions)
    {
        var sequenceBuilder = new SequenceBuilder(tokenizer, specialIds);
        var items = questions
            .Select(x =>
            {
                var sequence = sequenceBuilder.Build(state, x, layaConfig.MaxLength, layaConfig.HeadMaxLength);
                return new
                {
                    QuestionBase = x,
                    sequence.Ids, 
                    sequence.Markers
                };
            })
            .ToList();

        var n = items.Count;
        var l = items.Max(x => x.Ids.Length);
        var K = items.Max(x => x.Markers.Length);
        
        var inputIds = Enumerable
            .Repeat((long)specialIds.Pad, (int)(n * l))
            .ToArray();       
        var attention = new long[n * l];
        var markerPos = new long[n * K];
        var markerMask = new bool[n * K];
        var qtype = new long[n];
        
        var nTokens = 0;

        for (var i = 0; i < items.Count; i++)
        {
            var it = items[i];

            for (var j = 0; j < it.Ids.Length; j++)
            {
                inputIds[i * l + j] = it.Ids[j];
                attention[i * l + j] = 1;
            }

            nTokens += it.Ids.Length;

            for (var j = 0; j < it.Markers.Length; j++)
            {
                markerPos[i * K + j] = it.Markers[j];
                markerMask[i * K + j] = true;
            }

            qtype[i] = (long)it.QuestionBase.Kind;
        }
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(
                "input_ids",
                new DenseTensor<long>(inputIds, new[] { n, l })),
        
            NamedOnnxValue.CreateFromTensor(
                "attention_mask",
                new DenseTensor<long>(attention, new[] { n, l })),
        
            NamedOnnxValue.CreateFromTensor(
                "marker_pos",
                new DenseTensor<long>(markerPos, new[] { n, K })),
        
            NamedOnnxValue.CreateFromTensor(
                "marker_mask",
                new DenseTensor<bool>(markerMask, new[] { n, K })),
        
            NamedOnnxValue.CreateFromTensor(
                "qtype",
                new DenseTensor<long>(qtype, new[] { n }))
        };
        
        using var outs = session.Run(inputs);

        var logits = outs.First(x => x.Name == "logits").AsTensor<float>();
        var actProbs = outs.First(x => x.Name == "act_probs").AsTensor<float>();

        var nAct = session.OutputMetadata["act_probs"].Dimensions[1];

        var answers = new Dictionary<string, AnswerBase>();
        
        foreach (var item in items)
        {
            var key = item.QuestionBase.Key;
            
            var r = items.IndexOf(item);
            var start = r * K;
            var k = item.Markers.Length;
            var values = new float[k];
            
            var temp = layaConfig.TemperatureByOptions[Utils.TempBucket(item.QuestionBase.Kind, k)];

            for (int i = 0; i < k; i++)
            {
                values[i] = logits.GetValue(start + i) / temp;
            }

            var p = Utils.Softmax(values);
            var ext = actProbs.GetValue(r * nAct);

            if (item.QuestionBase.Kind == QuestionKind.Choice)
            {
                var choiceQuestion = item.QuestionBase as ChoiceQuestion;
                var keys = choiceQuestion.Criteria.Keys;
                var best = p.IndexOf(p.Max());
                var probabilities = keys.Select((kk, i) => new KeyValuePair<string, float>(kk, Utils.Round4(p[i]))).ToDictionary();
                var confidence = Utils.Round4(Utils.ConfidenceFromProbs(p));

                answers[key] = new ChoiceAnswer(keys.ElementAt(best), probabilities, confidence, ext);

            }
            else if (item.QuestionBase.Kind == QuestionKind.Score)
            {
                var scoreQuestion = item.QuestionBase as ScoreQuestion;
                
                var score = 0.0f;
                for (int i = 0; i < p.Length; i++)
                {
                    score += i * p[i];
                }

                score = Utils.Round4(score);
                var legend = scoreQuestion.Criteria.ToList()
                    .Select((c, i) => new KeyValuePair<string, string>(i.ToString(), c)).ToDictionary();
                var probabilities = p.Select((kk, i) => new KeyValuePair<string, float>(i.ToString(), Utils.Round4(kk))).ToDictionary();
                var confidence = Utils.Round4(Utils.ConfidenceFromProbs(p));
                
                answers[key] = new ScoreAnswer(score, legend, probabilities, confidence, ext);
            }
            else
            {
                answers[key] = new NoulAnswer(Utils.Round4(p[1]), ext);
            }

        }

        return new (answers, nTokens);
    }

    private static uint TokenToId(Tokenizer tokenizer, string token)
    {
        return tokenizer.Encode(token, false).First().Ids.First();
    }

    public void Dispose()
    {
        session.Dispose();
        tokenizer.Dispose();
    }
}

using System.Text.Json;
using LayaDotNet.Questions;
using Tokenizers.HuggingFace.Tokenizer;

namespace LayaDotNet;

public class SequenceBuilder(Tokenizer tokenizer, SpecialIds specialIds)
{
    public Sequence Build(string state, QuestionBase questionBase, int maxLength, int headMaxLength)
    {
        var opts = RenderOptions(questionBase);
        var encodedQuestion = $"{Utils.QuestionType(questionBase.Kind)} question: {Utils.Scrub(questionBase.Instructions, specialIds)}";

        var headIds = tokenizer.Encode(encodedQuestion, false).First().Ids;

        var optIds = opts.Select(x =>
        {
            var encodedOpt = " " + Utils.Scrub(x, specialIds);
            return tokenizer.Encode(encodedOpt, false).First().Ids
                .Take(48)
                .Prepend(specialIds.Mask)
                .ToArray();
        }).ToArray();

        var optIdsCount = optIds.Sum(x => x.Length);

        var budget = headMaxLength - optIdsCount;
        var seq = new List<uint> { specialIds.Cls };
        seq.AddRange(headIds.Take(Math.Max(8, budget)));
        seq.Add(specialIds.Sep);

        var markers = new List<int>();

        foreach (var o in optIds)
        {
            markers.Add(seq.Count);
            seq.AddRange(o);
        }

        seq.Add(specialIds.Sep);

        var room = Math.Max(0, maxLength - seq.Count - 1);
        var st = tokenizer.Encode(Utils.Scrub(JsonSerializer.Serialize(state, new JsonSerializerOptions
        {
            WriteIndented = false
        }), specialIds), false).First().Ids.Take(room);

        seq.AddRange(st);
        seq.Add(specialIds.Sep);

        return new Sequence(seq.Take(maxLength).ToArray(), markers.Where(x => x < maxLength).ToArray());
    }

    public static string[] RenderOptions(QuestionBase questionBase)
    {
        if (questionBase.Kind == QuestionKind.Choice)
        {
            var choiceQuestion = questionBase as ChoiceQuestion;

            return choiceQuestion!.Criteria
                .Select(kv => $"{kv.Key}: {kv.Value}")
                .ToArray();
        }

        if (questionBase.Kind == QuestionKind.Score)
        {
            var scoreQuestion = questionBase as ScoreQuestion;

            return scoreQuestion!.Criteria
                .Select((c, i) => $"level {i}: {c}")
                .ToArray();
        }

        return
        [
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        ];
    }
}

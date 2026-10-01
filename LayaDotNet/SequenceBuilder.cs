using System.Text.Json;
using LayaDotNet.Questions;
using Tokenizers.HuggingFace.Tokenizer;

namespace LayaDotNet;

public class SequenceBuilder(Tokenizer tokenizer, SpecialIds specialIds)
{
    public Sequence Build(string state, QuestionBase questionBase, int maxLength, int headMaxLength)
    {
        var opts = RenderOptions(questionBase);
        var encodedQuestion = $"{Utils.QuestionType(questionBase.Kind)} question: {Scrub(questionBase.Instructions)}";
        var headIds = tokenizer.Encode(encodedQuestion, false);

        var optIds = opts.Select(x =>
        {
            var encodedOpt = " " + Scrub(x);
            return tokenizer.Encode(encodedOpt, false).First().Ids.Take(48).Prepend(specialIds.Mask);
        }).ToList();
        
        var optIdsCount = optIds.Sum(x => x.Count());

        var budget = headMaxLength - optIdsCount;
        if (budget < 16)
        {
            // Too many / Too long options
        }
        
        var headIdsUint = headIds.Take(Math.Max(8, budget)).Select(x => x.Ids.First()).ToList();
        var seq = headIdsUint.Prepend(specialIds.Cls).Append(specialIds.Sep);

        var markers = new List<int>();
        
        foreach (var o in optIds)
        {
            var enumerable = seq as uint[] ?? seq.ToArray();
            
            markers.Add(enumerable.Count());
            seq = enumerable.Concat(o);
        }

        seq = seq.Append(specialIds.Sep);
        var room = Math.Max(0, maxLength - seq.Count() - 1);
        var st = tokenizer.Encode(Scrub(JsonSerializer.Serialize(state, new JsonSerializerOptions
        {
            WriteIndented = false
        })), false).First().Ids.Take(room);

        seq = seq.Concat(st);
        seq = seq.Append(specialIds.Sep);

        return new Sequence(seq.Take(maxLength).ToArray(), markers.Where(x => x < maxLength).ToArray());
    }

    private string Scrub(string s)
    {
        return string.Join(" ", s.Split(specialIds.MaskTok));
    }

    public static string[] RenderOptions(QuestionBase questionBase)
    {
        if (questionBase.Kind == QuestionKind.Choice)
        {
            var choiceQuestion = questionBase as ChoiceQuestion;

            return choiceQuestion.Criteria
                .Select(kv => kv.Value != null
                    ? $"{kv.Key}: {kv.Value}"
                    : kv.Key)
                .ToArray();
        }

        if (questionBase.Kind == QuestionKind.Score)
        {
            var scoreQuestion = questionBase as ScoreQuestion;

            return scoreQuestion.Criteria
                .Select((c, i) => $"level {i}: {c}")
                .ToArray();
        }

        return new[]
        {
            "false: no, the statement does not hold",
            "true: yes, the statement holds"
        };
    }
}
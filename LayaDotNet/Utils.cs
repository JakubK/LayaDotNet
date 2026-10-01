namespace LayaDotNet;

public static class Utils
{
    public static float[] Softmax(float[] z)
    {
        var zmax = z.Max();
        var e = z.Select(v => MathF.Exp(v - zmax)).ToList();
        var sum = e.Sum();
        return e.Select(x => x / sum).ToArray();
    }

    public static float ConfidenceFromProbs(float[] probs)
    {
        var k = probs.Length;
        if (k < 2) return 1;

        var ent = 0.0f;
        foreach (var prob in probs)
        {
            ent -= prob * MathF.Log(MathF.Max(prob, 1e-12f));
        }

        return 1 - ent / MathF.Log(k);
    }

    public static string SizeBucket(int k)
    {
        return k switch
        {
            <= 2 => "2",
            <= 5 => "3-5",
            <= 10 => "6-10",
            _ => "11+"
        };
    }
    
    public static string QuestionType(QuestionKind kind)
    {
        return kind switch
        {
            QuestionKind.Choice => "choice",
            QuestionKind.Score => "score",
            _ => "noul"
        };
    }

    public static string TempBucket(QuestionKind kind, int k)
    {
        return $"{QuestionType(kind)}:{SizeBucket(k)}";
    }

    public static float Round4(float number) => MathF.Round(number * 1e4f) / 1e4f;
}
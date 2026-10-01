using LayaDotNet.Answers;

namespace LayaDotNet;

public record SystemOneResponse(Dictionary<string, AnswerBase> Answers, int TokenUsage)
{
    public ChoiceAnswer Choice(string key) => (Answers[key] as ChoiceAnswer)!;
    public ScoreAnswer Score(string key) => (Answers[key] as ScoreAnswer)!;
    public NoulAnswer Noul(string key) => (Answers[key] as NoulAnswer)!;
}

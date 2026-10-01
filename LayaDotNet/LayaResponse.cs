using LayaDotNet.Answers;

namespace LayaDotNet;

public record LayaResponse(Dictionary<string, AnswerBase> Answers, int TokenUsage);

namespace LayaDotNet.Answers;

public record ScoreAnswer(float Score, Dictionary<string, string> Legend, Dictionary<string, float> Probabilities, float Confidence, float ActProbability) : AnswerBase(QuestionKind.Score, ActProbability);

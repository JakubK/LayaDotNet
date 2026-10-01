namespace LayaDotNet.Answers;

public record ChoiceAnswer(string Choice, Dictionary<string, float> Probabilities, float Confidence, float ActProbability) : AnswerBase(QuestionKind.Choice, ActProbability);
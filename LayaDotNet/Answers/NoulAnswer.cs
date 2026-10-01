namespace LayaDotNet.Answers;

public record NoulAnswer(float Noul, float ActProbability) : AnswerBase(QuestionKind.Noul, ActProbability);

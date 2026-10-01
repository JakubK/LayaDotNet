namespace LayaDotNet.Questions;

public record ChoiceQuestion(string Key, string Instructions, Dictionary<string, string> Criteria) : QuestionBase(Key, QuestionKind.Choice, Instructions);
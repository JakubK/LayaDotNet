namespace LayaDotNet.Questions;

public record NoulQuestion(string Key, string Instructions) : QuestionBase(Key, QuestionKind.Noul, Instructions);
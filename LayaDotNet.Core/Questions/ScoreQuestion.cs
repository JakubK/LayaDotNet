namespace LayaDotNet.Questions;

public record ScoreQuestion(string Key, string Instructions, IEnumerable<string> Criteria) : QuestionBase(Key, QuestionKind.Score, Instructions);
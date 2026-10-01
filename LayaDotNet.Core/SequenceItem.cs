using LayaDotNet.Questions;

namespace LayaDotNet;

public record SequenceItem(QuestionBase QuestionBase, uint[] Ids, int[] Markers);
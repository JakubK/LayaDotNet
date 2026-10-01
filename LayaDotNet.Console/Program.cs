using LayaDotNet;
using LayaDotNet.Answers;
using LayaDotNet.Questions;

var x = await Laya.LoadAsync();

var result = x.Predict("I need help with product which is not working as expected", [
    new ChoiceQuestion("department", "Which team should handle this ticket?", new ()
    {
        {"billing", "payments, refunds, invoices"},
        {"support", "product help and bugs"},
        {"sales", "new purchases"}
    }),
    new ScoreQuestion("urgency", "How urgent is this ticket?", ["not urgent", "somewhat urgent", "urgent", "critical"]),
    new NoulQuestion("churn_risk", "Is the customer likely to cancel or dispute?")
]);

var choice = result.Answers["department"] as ChoiceAnswer;
var score = result.Answers["urgency"] as ScoreAnswer;
var noul = result.Answers["churn_risk"] as NoulAnswer;

Console.WriteLine(choice!.Choice);
foreach (var kvp in choice.Probabilities)
{
    Console.WriteLine(kvp.Key + " => " + kvp.Value);
}
Console.WriteLine(score.Score);
Console.WriteLine(noul.Noul);
Console.WriteLine(result.TokenUsage);


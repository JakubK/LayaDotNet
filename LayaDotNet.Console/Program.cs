using LayaDotNet;
using LayaDotNet.Questions;

var laya = await Laya.LoadAsync();

var response = laya.Predict("I need help with product which is not working as expected", [
    new ChoiceQuestion("department", "Which team should handle this ticket?", new ()
    {
        {"billing", "payments, refunds, invoices"},
        {"support", "product help and bugs"},
        {"sales", "new purchases"}
    }),
    new ScoreQuestion("urgency", "How urgent is this ticket?", ["not urgent", "somewhat urgent", "urgent", "critical"]),
    new NoulQuestion("churn_risk", "Is the customer likely to cancel or dispute?")
]);

var choice = response.Choice("department");
var score = response.Score("urgency");
var noul = response.Noul("churn_risk");

Console.WriteLine(choice.Choice);
foreach (var kvp in choice.Probabilities)
{
    Console.WriteLine(kvp.Key + " => " + kvp.Value);
}
Console.WriteLine(score.Score);
Console.WriteLine(noul.Noul);
Console.WriteLine(response.TokenUsage);


using System.Diagnostics;
using LayaDotNet.Gpu;
using LayaDotNet.Questions;

var timer = new Stopwatch();
var laya = await Laya.LoadAsync();

timer.Start();

var response = laya.Predict("I need help with product which is not working as expected", [
    new ChoiceQuestion("department", "Which team should handle this ticket?", new ()
    {
        {"billing", "payments, refunds, invoices"},
        {"support", "product help and bugs"},
        {"sales", "new purchases and upgrades"}
    }),
    new ScoreQuestion("urgency", "How urgent is this ticket?", ["not urgent", "somewhat urgent", "urgent", "critical"]),
    new NoulQuestion("churn_risk", "Is the customer likely to cancel or dispute?")
]);

timer.Stop();

var department = response.Choice("department");
var urgency = response.Score("urgency");
var churnRisk = response.Noul("churn_risk");

Console.WriteLine(department.Choice);
foreach (var kvp in department.Probabilities)
{
    Console.WriteLine(kvp.Key + " => " + kvp.Value);
}
Console.WriteLine(urgency.Score);
Console.WriteLine(churnRisk.Noul);
Console.WriteLine(response.TokenUsage);

Console.WriteLine(timer.ElapsedMilliseconds + " ms");
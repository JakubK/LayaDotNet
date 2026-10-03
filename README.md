## LayaDotNet

Simple .NET library for running Laya System1 Decision Model inference locally from pointed .onnx files.
It lets you download onnx from hugging face (defaulting to receptron/laya) and run using CPU or GPU (separate package for each backend).

```csharp
using LayaDotNet.Gpu;
using LayaDotNet.Questions;

var laya = await Laya.LoadAsync();

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

var department = response.Choice("department");
var urgency = response.Score("urgency");
var churnRisk = response.Noul("churn_risk");
```

### Acknowledgements

- [receptron/laya Node.js library](https://github.com/receptron/laya)
- [receptron/laya onnx](https://huggingface.co/receptron/laya-onnx)
- [original laya model repo](https://github.com/NandhaKishorM/laya)

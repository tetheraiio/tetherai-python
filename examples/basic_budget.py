from tetherai import tether, BudgetExceededError

import litellm


@tether.enforce_budget(max_usd=0.05, trace_export="console")
def my_workflow():
    for i in range(100):
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Count to {i}"}]
        )
        print(f"Call {i+1}: {response.choices[0].message.content[:50]}...")


try:
    my_workflow()
except BudgetExceededError as e:
    print(f"\n🛑 Stopped! Spent ${e.spent_usd:.4f} of ${e.budget_usd:.2f} budget")

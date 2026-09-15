## Plan to Modify Wait Probability Based on Time of Day

The goal is to increase the probability of "waiting" (Action 4) when carbon intensity is likely high (nighttime) and decrease it when carbon intensity is likely low (daytime, noon), without directly using carbon intensity as an input feature.

We will achieve this by applying a bias to the logit of the "Wait" action in the `sample_action` method of `policy_gradient.py`. We will use the `cos_tod` (Cosine of Time of Day) which correlates with the daily solar cycle.

### Implementation Details:

1.  **Modify `src/model/policy-gradient-moe/policy_gradient.py`**:
    *   In the `sample_action` method (around line 408), after getting `logits`.
    *   Inject a bias term to the logit corresponding to the "Wait" action (Index 4).
    *   **Logic**:
        *   `cos_tod` is `1.0` at midnight (Night) and `-1.0` at noon (Day).
        *   We want higher wait probability at night (high carbon).
        *   We want lower wait probability at noon (low carbon).
        *   We will add `alpha * cos_tod` to `logits[:, 4]`.
        *   A proposed value for `alpha` is `5.0`.

### Proposed Code Change:

In `src/model/policy-gradient-moe/policy_gradient.py`:

```python
    def sample_action(self, data_input):
        # ... (existing code) ...
        
        # gating_logits
        logits = self.agent.forward(tensor_in, extra_input=extra_tensor)
        
        # --- START CHANGE ---
        # Modify Wait Logit (Index 4) based on Time of Day
        # cos_tod is 1 at midnight (Wait more), -1 at noon (Wait less)
        wait_bias_strength = 5.0
        # logits is shape [batch, num_actions], here [1, 5]
        # ensure cos_tod is a float
        logits[:, 4] += wait_bias_strength * float(cos_tod)
        # --- END CHANGE ---

        # softmax
        dist = Categorical(logits=logits)
        # ... (existing code) ...
```

This change strictly uses the time input (via `cos_tod`) and shapes the policy distribution as requested without altering the core model architecture or adding carbon inputs.

## Alternative Plan: Learn from Cyclic Time Features (No Prior)

Instead of manually injecting a bias (a prior) into the logits, we can enable the agent to **learn** the optimal waiting strategy by providing a better representation of time. The raw `tod` (Time of Day) input is non-cyclic (jumps from 24 to 0), making it difficult for the network to understand the continuity of night across midnight. By replacing `tod` with `sin_tod` and `cos_tod`, we give the agent a continuous, cyclic view of the day, allowing it to easily correlate "nighttime" with "high carbon penalty" and learn to wait appropriately.

### Implementation Details:

1.  **Modify `src/model/policy-gradient-moe/policy_gradient.py` initialization**:
    *   Change `extra_dim` from `1` to `2` (around line 184).

2.  **Modify `sample_action` method**:
    *   Update the `extra_tensor` construction (around line 405) to use the calculated `sin_tod` and `cos_tod`.
    *   `extra_tensor = torch.tensor([[sin_tod, cos_tod]], dtype=torch.float32).to(self.DEVICE)`
    *   Do **not** apply the manual logit bias mentioned in the previous plan.

### Proposed Code Change:

In `src/model/policy-gradient-moe/policy_gradient.py`:

```python
        # ... inside __init__ ...
        # Enhanced to use Sin/Cos Cyclic Encoding for Time of Day
        extra_dim = 2 # Changed from 1 to 2
        
        # ... (rest of init) ...

        # ... inside sample_action(self, data_input) ...
        # Use Sin/Cos Cyclic Encoding
        # Normalize 0-24h to 0-2pi
        norm_tod = (tod / 24.0) * 2 * np.pi
        sin_tod = np.sin(norm_tod)
        cos_tod = np.cos(norm_tod)
        
        # Pass sin/cos as inputs instead of raw tod
        extra_tensor = torch.tensor([[sin_tod, cos_tod]], dtype=torch.float32).to(self.DEVICE)
        
        # gating_logits
        logits = self.agent.forward(tensor_in, extra_input=extra_tensor)
        
        # No manual bias addition here. Let the agent learn from the inputs.
```

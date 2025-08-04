# --- Setup ---
from torch import nn
import torch
import copy
from axolotl.utils.optimizers.adamw_bayes import AdamWBayes


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def main():
    # Create two identical models that start from the same weights
    model_bayes = SimpleNet()
    model_standard = SimpleNet()
    model_standard.load_state_dict(
        model_bayes.state_dict()
    )  # Ensure they are identical

    # Store the original weights for later comparison
    original_weights = copy.deepcopy(model_bayes.state_dict())

    # --- Optimizers ---
    print("🚀 Setting up optimizers...")
    optimizer_bayes = AdamWBayes(
        model_bayes.parameters(),
        lr=0.01,
        weight_decay=0.1,  # A high value to make the effect obvious
        clone_params_for_reference=True,  # The key feature!
    )
    optimizer_standard = torch.optim.AdamW(
        model_standard.parameters(),
        lr=0.01,
        weight_decay=0.1,  # Same hyperparameters for a fair comparison
    )
    # --- Dummy Data and Training ---
    data = torch.randn(16, 10)
    labels = torch.randn(16, 2)
    criterion = nn.MSELoss()

    print("🏋️  Training for a few steps...")
    for _ in range(20):
        # Train AdamWBayes
        optimizer_bayes.zero_grad()
        loss_bayes = criterion(model_bayes(data), labels)
        loss_bayes.backward()
        optimizer_bayes.step()

        # Train standard AdamW
        optimizer_standard.zero_grad()
        loss_standard = criterion(model_standard(data), labels)
        loss_standard.backward()
        optimizer_standard.step()

    print("\n✅ Training complete.")
    print("-" * 50)

    # --- Verification ---
    print("🔎 Comparing final weights to original weights...")

    # Calculate L2 distance from original weights for a specific layer
    original_l1_weights = original_weights["layer1.weight"]
    bayes_l1_weights = model_bayes.state_dict()["layer1.weight"]
    standard_l1_weights = model_standard.state_dict()["layer1.weight"]

    dist_bayes = torch.norm(bayes_l1_weights - original_l1_weights)
    dist_standard = torch.norm(standard_l1_weights - original_l1_weights)

    print(f"\nDistance from original weights (Standard AdamW):  {dist_standard:.4f}")
    print(f"Distance from original weights (AdamWBayes):     {dist_bayes:.4f}")

    if dist_bayes < dist_standard:
        print(
            "\n🎉 Success! AdamWBayes model stayed significantly closer to the original weights."
        )
    else:
        print("\nSomething went wrong. The effect was not observed.")


# %%
if __name__ == "__main__":
    main()
    print("🚀 Running the AdamWBayes optimizer test...")

import torch
import math
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class AdamWBayes(Optimizer):  # Inherit from the base Optimizer for clarity
    def __init__(
        self,
        params,
        reference_params: list[torch.Tensor] = None,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        clone_params_for_reference: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if clone_params_for_reference and reference_params is not None:
            raise ValueError(
                "You cannot specify both `reference_params` and set "
                "`clone_params_for_reference=True`. Please choose one."
            )

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(params, defaults)

        # Associate reference_params with their corresponding param_groups
        if clone_params_for_reference:
            # Loop through the now-existing groups and clone their params
            for group in self.param_groups:
                group["reference_params"] = [
                    p.detach().clone() for p in group["params"]
                ]
        elif reference_params is not None:
            # Logic to inject user-provided reference_params
            ref_params_list = list(reference_params)
            main_param_count = sum(len(group["params"]) for group in self.param_groups)
            if main_param_count != len(ref_params_list):
                raise ValueError(
                    f"Number of main parameters ({main_param_count}) does not match "
                    f"number of reference parameters ({len(ref_params_list)})"
                )

            ref_param_idx = 0
            for group in self.param_groups:
                num_params_in_group = len(group["params"])
                group["reference_params"] = ref_params_list[
                    ref_param_idx : ref_param_idx + num_params_in_group
                ]
                ref_param_idx += num_params_in_group
        else:
            # Default case: no reference params
            for group in self.param_groups:
                group["reference_params"] = [None] * len(group["params"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            reference_params_with_grad = []
            beta1, beta2 = group["betas"]

            ## REFACTOR: Consolidate loop to avoid duplication
            for p, ref_p in zip(group["params"], group["reference_params"]):
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                grads.append(p.grad)
                reference_params_with_grad.append(ref_p)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            # Call the functional API with the correctly filtered lists
            self.adamw_bayes_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                reference_params=reference_params_with_grad,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    @staticmethod
    def adamw_bayes_update(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        *,
        reference_params: List[Optional[torch.Tensor]],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
    ):
        ## REFACTOR: Consolidate logic into a single loop
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i].item()
            reference_param = reference_params[i]

            # Decoupled weight decay logic
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)
                # This is the only part that depends on reference_param
                if reference_param is not None:
                    param.add_(reference_param, alpha=lr * weight_decay)

            # Core Adam algorithm
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(
                    eps
                )
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)

import torch

class LeakyIandF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_currents, threshold, scaling, mult_factor, alpha):
        ctx.set_materialize_grads(False)
        ctx.threshold = threshold
        ctx.scaling = scaling
        ctx.mult_factor = mult_factor
        ctx.alpha = alpha

        membrane_potentials = input_currents.clone()
        output_spikes = torch.zeros(membrane_potentials.shape,
                                    dtype=torch.bool,
                                    device=membrane_potentials.device)

        for step in range(input_currents.shape[-1]):
            pots = membrane_potentials[..., step] * scaling
            if step > 0:
                pots += mult_factor * membrane_potentials[..., step-1] + scaling * prev_currents

            output_spikes[..., step] = pots > threshold
            pots[output_spikes[..., step]] = 0
            membrane_potentials[..., step] = pots

            prev_currents = input_currents[..., step]
            prev_currents[output_spikes[..., step]] = 0

        print(f"LeakyIandF: {membrane_potentials.min()}, {membrane_potentials.max()}, {membrane_potentials.mean()}, {membrane_potentials.std()}")

        ctx.save_for_backward(membrane_potentials, output_spikes)

        return output_spikes.float()

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None

        membrane_potentials, output_spikes = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            threshold_gating = 0.5 / ctx.alpha

            spike_grads = grad_output
            mask = torch.abs(membrane_potentials - ctx.threshold) > ctx.alpha
            spike_grads[mask] = 0
            spike_grads *= threshold_gating

            potential_grads = spike_grads
            grad_input = potential_grads.clone()
            for step in range(potential_grads.shape[-1]-2, -1, -1):
                pot_grad = potential_grads[..., step]
                next_grad = potential_grads[..., step+1]
                mask = torch.logical_not(output_spikes[..., step])
                pot_grad[mask] += ctx.mult_factor * next_grad[mask]
                potential_grads[..., step] = pot_grad
                next_grad[output_spikes[..., step]] = 0
                grad_input[..., step] = (pot_grad + next_grad)
            grad_input = grad_input * ctx.scaling

        return grad_input, None, None, None, None

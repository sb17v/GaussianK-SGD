""" Layer-wise adaptive rate scaling for SGD in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eeta=0.0001, epsilon=1e-5, max_epoch=200):
       
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eeta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        #Initialize the variables required
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._eeta = eeta
        self._max_epoch = max_epoch

        #Initialize the default state dictionary
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eeta=eeta, epsilon=epsilon, max_epoch=max_epoch)

        #Initialize the parenr constructor with the default state dictionary
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            learning_rate = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            eeta = group['eeta']
            epsilon = group['epsilon']
            max_epoch = group['max_epoch']

            average_lr = 0
            iteration = 0
            for param in group['params']:
                if param.grad is None:
                    continue

                param_state = self.state[param]

                grad = param.grad.data
                weight = param.data

                weight_norm = torch.norm(weight, p = 2)
                grad_norm = torch.norm(grad, p = 2)
               
                #Step 1: Scaling Factor = eeta
                trust_ratio = torch.where(
                    torch.gt(weight_norm, torch.zeros_like(weight_norm)), 
                    torch.where(torch.gt(grad_norm, torch.zeros_like(weight_norm)), 
                    (eeta * weight_norm/(grad_norm + weight_decay * weight_norm + epsilon)), 
                    torch.ones_like(weight_norm)),
                    torch.ones_like(weight_norm))

                trust_ratio = trust_ratio.clamp_(0.0, 50.0)
          
                #Step 2: Scaled LR for each layer
                scaled_lr = learning_rate * trust_ratio

                #For Debugging purpose
                average_lr += scaled_lr

                #Step 3: Update the gradients
                decayed_grad = grad  + weight_decay * weight
                decayed_grad = decayed_grad.clamp_(-10.0, 10.0)

                #Create a acceleration buffer if not available and initialize it to 0
                if 'acceleration' not in param_state:
                    acc_buf = param_state['acceleration'] = \
                            torch.ones_like(weight)
                else:
                    acc_buf = param_state['acceleration']

                #Step 4: Calculate acceleration term 
                #temp_buf = torch.zeros_like(weight)
                #temp_buf.mul_(scaled_lr, alpha = decayed_grad)

                acc_buf.mul_(momentum)
                acc_buf.add_(decayed_grad, alpha = scaled_lr)

                param_state['acceleration'] = acc_buf

                #Step 5: Perform weight update
                param.data.add_(-acc_buf)
                #param.data.add_(-self._lr * grad)
        return loss



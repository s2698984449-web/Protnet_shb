#  #!/usr/bin/env python3
#
# import traceback
# from learn2learn.algorithms.base_learner import BaseLearner
# from learn2learn.utils import clone_module, update_module
# import torch
# from torch.distributions import StudentT
# from learn2learn.utils import clone_module, update_module
# from torch.autograd import grad
# import math
#
#
# def maml_update(model, lr, grads=None):
#
#     if grads is not None:
#         params = list(model.parameters())
#         if not len(grads) == len(list(params)):
#             msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
#             msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
#             print(msg)
#         for p, g in zip(params, grads):
#             if g is not None:
#                 p.update = - lr * g
#     return update_module(model)
#
# class MAML(BaseLearner):
#
#     def __init__(self,
#                  model,
#                  lr,
#                  first_order=False,
#                  allow_unused=None,
#                  allow_nograd=False):
#         super(MAML, self).__init__()
#         self.module = model
#         self.lr = lr
#         self.first_order = first_order
#         self.allow_nograd = allow_nograd
#         if allow_unused is None:
#             allow_unused = allow_nograd
#         self.allow_unused = allow_unused
#
#     def forward(self, *args, **kwargs):
#         return self.module(*args, **kwargs)
#
#     def adapt(self,
#               loss,
#               first_order=None,
#               allow_unused=None,
#               allow_nograd=None):
#
#         if first_order is None:
#             first_order = self.first_order
#         if allow_unused is None:
#             allow_unused = self.allow_unused
#         if allow_nograd is None:
#             allow_nograd = self.allow_nograd
#         second_order = not first_order
#
#         if allow_nograd:
#             # Compute relevant gradients
#             diff_params = [p for p in self.module.parameters() if p.requires_grad]
#             grad_params = grad(loss,
#                                diff_params,
#                                retain_graph=second_order,
#                                create_graph=second_order,
#                                allow_unused=allow_unused)
#             gradients = []
#             grad_counter = 0
#
#             # Handles gradients for non-differentiable parameters
#             for param in self.module.parameters():
#                 if param.requires_grad:
#                     gradient = grad_params[grad_counter]
#                     grad_counter += 1
#                 else:
#                     gradient = None
#                 gradients.append(gradient)
#         else:
#             try:
#                 gradients = grad(loss,
#                                  self.module.parameters(),
#                                  retain_graph=second_order,
#                                  create_graph=second_order,
#                                  allow_unused=allow_unused)
#             except RuntimeError:
#                 traceback.print_exc()
#                 print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')
#
#         # Update the module
#         self.module = maml_update(self.module, self.lr, gradients)
#
#     def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
#
#         if first_order is None:
#             first_order = self.first_order
#         if allow_unused is None:
#             allow_unused = self.allow_unused
#         if allow_nograd is None:
#             allow_nograd = self.allow_nograd
#         return MAML(clone_module(self.module),
#                     lr=self.lr,
#                     first_order=first_order,
#                     allow_unused=allow_unused,
#                     allow_nograd=allow_nograd)

#!/usr/bin/env python3

import traceback
from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, update_module
import torch
from torch.distributions import StudentT
from learn2learn.utils import clone_module, update_module
from torch.autograd import grad
import math

# 辅助函数：计算梯度的范数
def compute_grad_norm(gradients):
    grad_norms = []
    for grad in gradients:
        if grad is not None:
            grad_norm = grad.pow(2).sum().sqrt().item()  # 计算L2范数
        else:
            grad_norm = 0.0
        grad_norms.append(grad_norm)
    return grad_norms

def maml_update(model, lr, grads=None, task_weights=None):

    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(params):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g, w in zip(params, grads, task_weights):  # 加入task_weights
            if g is not None:
                p.update = - lr * w * g  # 用任务权重调整梯度更新
    return update_module(model)

class MAML(BaseLearner):

    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        # 计算梯度
        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # 计算每个任务的梯度范数（根据任务权重调整）
        grad_norms = compute_grad_norm(gradients)
        total_norm = sum(grad_norms)  # 所有任务梯度范数的总和
        task_weights = [grad_norm / total_norm if total_norm > 0 else 0.0 for grad_norm in grad_norms]

        # 使用梯度加权更新
        self.module = maml_update(self.module, self.lr, gradients, task_weights)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)


import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import process_dataset
from dataset import make_dataset, make_data_loader
from model import make_model
from module import save, process_control, makedir_exist_ok, Controller

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    # non_blocking = True
    # a = torch.randn(10000000, device='cuda')
    # print(a)
    # s = time.time()
    # a = a.to('cpu', non_blocking=non_blocking)
    # torch.cuda.synchronize()
    # a *= 2
    # print(a)
    # e = time.time()
    # print(e - s)

    # # Define a callback function
    # def callback(status):
    #     print("Transfer Complete")
    #
    # # Create a tensor on the GPU
    # tensor_cuda = torch.randn(10, 10, device='cuda')
    #
    # # Create a custom stream
    # stream = torch.cuda.Stream()
    #
    # # Perform operations on the stream
    # with torch.cuda.stream(stream):
    #     # Transfer the tensor to CPU
    #     tensor_cpu = tensor_cuda.to('cpu', non_blocking=True)
    #
    #     # Add a callback to the stream
    #     stream.add_callback(callback)

    # Initialize an empty tensor
    # empty_tensor = torch.Tensor()
    #
    # # Example tensors to concatenate
    # tensor1 = torch.randn(3, 4)  # Example tensor of size [3, 4]
    # tensor2 = torch.randn(3, 4)  # Another tensor of size [3, 4]
    #
    # # Concatenate tensors along a specific dimension (e.g., dimension 0)
    # concatenated_tensor = torch.cat((empty_tensor, tensor1, tensor2), dim=0)
    #
    # print(concatenated_tensor.size())

    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # stream = torch.cuda.current_stream()
    # print(stream.query())
    # x = torch.rand(32, 256, 220, 220).cuda()
    #
    # t = (x.min() - x.max()).to(torch.device("cpu"), non_blocking=True)
    # print(stream.query())  # False - work not done yet
    # stream.synchronize()  # wait for stream to finish the work
    # print(t)
    #
    # # time.sleep(2.)
    # print(stream.query())  # True - work done
    # print(t)

    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # stream = torch.cuda.Stream()
    # x = torch.rand(32, 256, 220, 220).cuda()
    #
    # with torch.cuda.stream(stream):
    #     y = x.to('cpu', non_blocking=True)
    # print(y.min(), y.max())
    # print(stream.query())  # False - work not done yet
    # stream.synchronize()  # wait for stream to finish the work
    # print(y.min(), y.max())
    # print(stream.query())  # True - work done
    # print(y.min(), y.max())

    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # event = torch.cuda.Event()
    # x = torch.rand(32, 256, 220, 220).cuda()
    # print(event.query())
    # y = x.to('cpu', non_blocking=True)
    # event.record()
    # print(y.min(), y.max())
    # print(event.query())  # False - work not done yet
    # event.synchronize()  # wait for stream to finish the work
    # print(y.min(), y.max())
    # print(event.query())  # True - work done
    # print(y.min(), y.max())

    # model = nn.Linear(100, 10)
    # print(model.forward)
    # print(model.backward)

    # b_grad = []
    #
    # def backward_hook(grad):
    #     b_grad.append(grad)
    #     return
    #
    # a = torch.randn(10, 20, 3, 3)
    #
    # model = nn.Conv2d(20, 2, 3, 1, 1)
    # b = model(a)
    # # b.requires_grad_(True)
    # b.register_hook(backward_hook)
    # loss = b.pow(2).sum()
    # loss.backward()
    # print(model.weight.grad.abs().sum())
    # print(model.bias.grad.abs().sum())
    # model.zero_grad()
    # weight_grad = torch.nn.grad.conv2d_weight(a, model.weight.size(), b_grad[0], stride=model.stride, padding=model.padding,
    #                                       dilation=model.dilation, groups=model.groups)
    # bias_grad = b_grad[0].sum(dim=[0, 2, 3])
    # print(weight_grad.abs().sum())
    # print(bias_grad.abs().sum())

    # b_grad = []
    #
    # def backward_hook(grad):
    #     b_grad.append(grad)
    #     return
    #
    # a = torch.randn(10, 20, 3, 3)
    #
    # model = nn.BatchNorm2d(20)
    # b = model(a)
    # # b.requires_grad_(True)
    # b.register_hook(backward_hook)
    # loss = (b - torch.randn(b.size())).pow(2).sum()
    # loss.backward()
    # print(model.weight.grad.size())
    # print(model.weight.grad.abs().sum())
    # print(model.bias.grad.size())
    # print(model.bias.grad.abs().sum())
    # model.zero_grad()
    # normalized = torch.nn.functional.batch_norm(a, None, None, weight=None, bias=None, training=True,
    #                                         momentum=model.momentum, eps=model.eps)
    # weight_grad = (b_grad[0] * normalized).sum(dim=[0, 2, 3])
    # bias_grad = b_grad[0].sum(dim=[0, 2, 3])
    # print(weight_grad.size())
    # print(weight_grad.abs().sum())
    # print(bias_grad.size())
    # print(bias_grad.abs().sum())
    # exit()

    # b_grad = []
    #
    # def backward_hook(grad):
    #     b_grad.append(grad)
    #     return
    #
    # a = torch.randn(10, 100, 20)
    #
    # model = nn.LayerNorm(20)
    # b = model(a)
    # # b.requires_grad_(True)
    # b.register_hook(backward_hook)
    # loss = (b - torch.randn(b.size())).pow(2).sum()
    # loss.backward()
    # print(model.weight.grad.size())
    # print(model.weight.grad.abs().sum())
    # print(model.bias.grad.size())
    # print(model.bias.grad.abs().sum())
    # model.zero_grad()
    # normalized = torch.nn.functional.layer_norm(a, model.normalized_shape, weight=None, bias=None, eps=model.eps)
    # weight_grad = (b_grad[0] * normalized).sum(dim=[0, 1])
    # bias_grad = b_grad[0].sum(dim=[0, 1])
    # print(weight_grad.size())
    # print(weight_grad.abs().sum())
    # print(bias_grad.size())
    # print(bias_grad.abs().sum())
    # exit()
    # import time
    # event_1 = torch.cuda.Event()
    # event_2 = torch.cuda.Event()
    # stream_1 = torch.cuda.Stream()
    # stream_2 = torch.cuda.Stream()
    #
    # N = 100
    # non_blocking = False
    # x_1 = torch.randn(N, 4096, 1024, device='cuda')
    # x_2 = torch.randn(N, 4096, 1024, device='cuda')
    # print(x_1.sum())
    # print(x_2.sum())
    #
    # x_1_cpu = torch.zeros(N, 4096, 1024)  # Pre-allocated on CPU
    # x_2_cpu = torch.zeros(N, 4096, 1024)
    #
    # s = time.time()
    # # with torch.cuda.stream(stream_1):
    # #     # x_1_cpu.copy_(x_1.to('cpu', non_blocking=non_blocking))
    # #     x_1_cpu.copy_(x_1)
    # #     # x_1_cpu = x_1.to('cpu', non_blocking=non_blocking)
    # #     event_1.record()
    # x_1_cpu.copy_(x_1, non_blocking=True)
    # event_1.record()
    #
    # print(time.time() - s)
    # # with torch.cuda.stream(stream_2):
    # #     x_2_cpu.copy_(x_2)
    #     # x_2_cpu = x_2.to('cpu', non_blocking=non_blocking)
    #     # event_2.record()
    # x_2_cpu.copy_(x_2, non_blocking=True)
    # event_2.record()
    # print(time.time() - s)
    #
    # # x_3 = torch.randn(N, 4096, 1024, device='cuda')
    # # time.sleep(1)
    # print(time.time() - s)
    # event_1.synchronize()
    # print(time.time() - s)
    # event_2.synchronize()
    # print(time.time() - s)
    #
    # print(x_1_cpu.sum())
    # print(x_2_cpu.sum())
    #
    # weight_grad = []

    # def backward_hook(tensor):
    #     print(tensor.grad)
    #     # weight_grad.append(grad)
    #     del tensor.grad
    #     return
    # x = torch.randn(100, 64)
    # model = nn.Linear(64, 20)
    # model.weight.register_post_accumulate_grad_hook(backward_hook)
    # model.bias.register_post_accumulate_grad_hook(backward_hook)
    #
    # y = model(x)
    # loss = y.pow(2).sum()
    # loss.backward()
    # print(model.weight.grad)
    # print(model.bias.grad)
    # exit()

    # import time
    # import threading
    # event_1 = torch.cuda.Event()
    # event_2 = torch.cuda.Event()
    # event = {'1': event_1, '2': event_2}
    # stream_1 = torch.cuda.Stream()
    # stream_2 = torch.cuda.Stream()
    # # threading.Thread(target=offload_to_cpu, args=(cuda_tensor,)).start()
    #
    # N = 100
    # non_blocking = True
    # x_1 = torch.randn(N, 4096, 512, device='cuda')
    # x_1.grad = torch.randn(N, 4096, 512, device='cuda')
    # x_2 = torch.randn(N, 4096, 512, device='cuda')
    # x_2.grad = torch.randn(N, 4096, 512, device='cuda')
    # print(x_1.grad.sum())
    # print(x_2.grad.sum())
    #
    # x_1_cpu = torch.zeros(N, 4096, 512)  # Pre-allocated on CPU
    # x_2_cpu = torch.zeros(N, 4096, 512)
    # x_cpu = {'1': x_1_cpu, '2': x_2_cpu}
    #
    # def offload_to_cpu(cuda_tensor, name):
    #     x_cpu[name].copy_(cuda_tensor.grad, non_blocking=non_blocking)
    #     # cuda_tensor.grad = None
    #     # x_cpu[name] = cuda_tensor.to('cpu', non_blocking=non_blocking)
    #     event[name].record()
    #     del cuda_tensor.grad
    #     event[name].synchronize()
    #     print('{} offload {}'.format(name, x_cpu[name].sum()), cuda_tensor.grad)
    #     return
    # s = time.time()
    # # with torch.cuda.stream(stream_1):
    # #     x_1_cpu.copy_(x_1.to('cpu', non_blocking=non_blocking))
    # #     # x_1_cpu.copy_(x_1)
    # #     # x_1_cpu = x_1.to('cpu', non_blocking=non_blocking)
    # #     event_1.record()
    # # x_1_cpu.copy_(x_1, non_blocking=non_blocking)
    # # event_1.record()
    #
    # threading.Thread(target=offload_to_cpu, args=(x_1, '1')).start()
    # print(time.time() - s)
    # with torch.cuda.stream(stream_2):
    #     x_2_cpu.copy_(x_2.to('cpu', non_blocking=non_blocking))
    # x_2_cpu.copy_(x_2)
    # x_2_cpu = x_2.to('cpu', non_blocking=non_blocking)
    # event_2.record()
    # x_2_cpu.copy_(x_2, non_blocking=True)
    # event_2.record()

    # threading.Thread(target=offload_to_cpu, args=(x_2, '2')).start()
    # print(time.time() - s)
    #
    # # x_3 = torch.randn(N, 4096, 1024, device='cuda')
    # # time.sleep(1)
    # print(time.time() - s)
    # event_1.synchronize()
    # print(time.time() - s)
    # event_2.synchronize()
    # print(time.time() - s)
    #
    # print(torch.cuda.memory_allocated())
    #
    # print(x_1_cpu.sum())
    # print(x_2_cpu.sum())
    #
    # time.sleep(3)
    # print(x_1_cpu.sum())
    # print(x_2_cpu.sum())
    # print(x_1.grad)
    # print(x_2.grad)

    def backward_hook(grad):
        print(grad)
        return

    N = 10
    d = 3
    x = torch.randn(N, d)
    y = torch.randn(N, d)
    linear_1 = nn.Linear(d, d, bias=False)
    linear_2 = nn.Linear(d, d, bias=False)
    # linear_3 = nn.Linear(d, d)
    # linear_4 = nn.Linear(d, d)

    x.requires_grad_(True)
    x.retain_grad()
    # x.register_hook(backward_hook)
    x_1 = linear_1(x) + x
    x_1.retain_grad()
    x_1.requires_grad_(True)
    # x_1.register_hook(backward_hook)
    x_2 = linear_2(x_1) + x_1
    x_2.requires_grad_(True)
    x_2.retain_grad()
    # x_2.register_hook(backward_hook)
    loss = (y - x_2).pow(2).sum()
    loss.backward()
    x_grad = x.grad
    x_1_grad = x_1.grad
    w_1_grad = linear_1.weight.grad
    x_2_grad = x_2.grad
    w_2_grad = linear_2.weight.grad

    # print('x_2', x_2)

    print('x_grad', x_grad)
    print('x_1_grad', x_1_grad)
    print('w_1_grad', w_1_grad)
    print('x_2_grad', x_2_grad)
    print('w_2_grad', w_2_grad)

    x.grad = None
    x_1.grad = None
    linear_1.weight.grad = None
    x_2.grad = None
    linear_2.weight.grad = None

    h = x.detach()

    new_x = x.detach()
    new_x.requires_grad_(True)
    h_1 = linear_1(new_x)

    new_x_1 = (h + h_1).detach()
    new_x_1.requires_grad_(True)
    h_2 = linear_2(new_x_1)

    all_h = h + h_1 + h_2
    all_h = all_h.detach()
    all_h.requires_grad_(True)
    loss = (y - all_h).pow(2).sum()
    loss.backward()
    # print('all_h', all_h)
    print('all_h_grad', all_h.grad)

    h_2.backward(all_h.grad)
    print('new_w_2_grad', linear_2.weight.grad)
    print('new_x_1_grad', new_x_1.grad)

    h_1.backward(all_h.grad)
    print('new_w_1_grad', linear_1.weight.grad)
    print('new_x_grad', new_x.grad)

    print('acc_new_x_1_grad', new_x_1.grad + all_h.grad)
    added_grad = (new_x.t() @ new_x_1.grad).t()

    print('acc_new_w_1_grad', linear_1.weight.grad + added_grad)

    return

    # def make_hook(self, learner_name, learner, hook, signal):
    #
    #     def _forward_hook(learner_name, learner, hook, signal, backward_hook, module, input, output):
    #         if self.model_gpu.training:
    #             signal['forward'][learner_name] = learner.forward.remote(input[0])
    #         output.requires_grad_(True)
    #         output.register_hook(backward_hook)
    #         return
    #
    #     def _backward_hook(learner_name, learner, signal, grad):
    #         if self.model_gpu.training:
    #             signal['backward'][learner_name] = learner.backward.remote(grad)
    #         return
    #
    #     backward_hook = partial(_backward_hook, learner_name, learner, signal)
    #     forward_hook = partial(_forward_hook, learner_name, learner, hook, backward_hook, signal)
    #
    #     return forward_hook, backward_hook


if __name__ == "__main__":
    main()

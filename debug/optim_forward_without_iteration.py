import torch
from torch.autograd import Function
from torch.nn import ReLU
import triton
import triton.language as tl

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_D': 128}, num_warps = 4, num_stages = 3),
            triton.Config({'BLOCK_SIZE_D': 256}, num_warps = 4, num_stages = 3),
            triton.Config({'BLOCK_SIZE_D': 512}, num_warps = 8, num_stages = 4),
            triton.Config({'BLOCK_SIZE_D': 1024}, num_warps = 8, num_stages = 4),
            triton.Config({'BLOCK_SIZE_D': 2048}, num_warps = 8, num_stages = 4),
        ],
        key=['D']
)
@triton.jit
def gradient_w_kernel(index_ptr, grad_ptr, delta_ptr, x_ptr,
                      D, N_INDEX,
                      stride_index_dim0, stride_index_dim1,
                      stride_grad_d, stride_grad_v,
                      stride_delta_b, stride_delta_v,
                      stride_x_b, stride_x_l, stride_x_d,
                      BLOCK_SIZE_D: tl.constexpr):
    pid = tl.program_id(axis=0)

    start_point = pid * BLOCK_SIZE_D

    offsets = start_point + tl.arange(0, BLOCK_SIZE_D)

    mask = offsets < D

    for i in range(N_INDEX):
        b = tl.load(index_ptr + (i * stride_index_dim0) + 0 * stride_index_dim1)
        v = tl.load(index_ptr + (i * stride_index_dim0) + 1 * stride_index_dim1)
        lmax = tl.load(index_ptr + (i * stride_index_dim0) + 2 * stride_index_dim1)

        delta = tl.load(delta_ptr + (b * stride_delta_b + v * stride_delta_v))
        x = tl.load(x_ptr + (b * stride_x_b + lmax * stride_x_l + offsets * stride_x_d), mask=mask)
        grad = tl.load(grad_ptr + (offsets * stride_grad_d + v * stride_grad_v), mask=mask)

        grad += delta * x

        tl.store(grad_ptr + (offsets * stride_grad_d + v * stride_grad_v), grad, mask=mask)


def compute_gradient_w(index, delta, x, grad_weight):
    N_INDEX = index.shape[0]
    D = x.shape[2]

    grid = lambda META: (triton.cdiv(D, META['BLOCK_SIZE_D']),)
    gradient_w_kernel[grid](index, grad_weight, delta, x,
                            D, N_INDEX,
                            index.stride(0), index.stride(1),
                            grad_weight.stride(0), grad_weight.stride(1),
                            delta.stride(0), delta.stride(1),
                            x.stride(0), x.stride(1), x.stride(2))

    return grad_weight.clone().detach().requires_grad_(True)

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_D': 128}, num_warps = 4, num_stages = 3),
            triton.Config({'BLOCK_SIZE_D': 256}, num_warps = 4, num_stages = 3),
            triton.Config({'BLOCK_SIZE_D': 512}, num_warps = 8, num_stages = 4),
            triton.Config({'BLOCK_SIZE_D': 1024}, num_warps = 8, num_stages = 4),
            triton.Config({'BLOCK_SIZE_D': 2048}, num_warps = 8, num_stages = 4),
        ],
        key=['D']
)
@triton.jit
def gradient_x_kernel(index_ptr, grad_ptr, delta_ptr, w_ptr,
                      D, N_INDEX,
                      stride_index_dim0, stride_index_dim1,
                      stride_grad_b, stride_grad_lmax, stride_grad_d,
                      stride_delta_b, stride_delta_v,
                      stride_w_d, stride_w_v,
                      BLOCK_SIZE_D: tl.constexpr):
    pid = tl.program_id(axis=0)

    start_point = pid * BLOCK_SIZE_D

    offsets = start_point + tl.arange(0, BLOCK_SIZE_D)

    mask = offsets < D

    for i in range(N_INDEX):
        b = tl.load(index_ptr + (i * stride_index_dim0) + 0 * stride_index_dim1)
        v = tl.load(index_ptr + (i * stride_index_dim0) + 1 * stride_index_dim1)
        lmax = tl.load(index_ptr + (i * stride_index_dim0) + 2 * stride_index_dim1)

        delta = tl.load(delta_ptr + (b * stride_delta_b + v * stride_delta_v))
        w = tl.load(w_ptr + (offsets * stride_w_d + v * stride_w_v), mask=mask)

        grad = tl.load(grad_ptr + (b * stride_grad_b + lmax * stride_grad_lmax + offsets * stride_grad_d), mask=mask)

        grad += delta * w

        tl.store(grad_ptr + (b * stride_grad_b + lmax * stride_grad_lmax + offsets * stride_grad_d), grad, mask=mask)


def compute_gradient_x(index, delta, w, grad_x):
    N_INDEX = index.shape[0]
    D = w.shape[0]

    grid = lambda META: (triton.cdiv(D, META['BLOCK_SIZE_D']),)
    gradient_x_kernel[grid](index, grad_x, delta, w,
                            D, N_INDEX,
                            index.stride(0), index.stride(1),
                            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
                            delta.stride(0), delta.stride(1),
                            w.stride(0), w.stride(1))

    return grad_x.clone().detach().requires_grad_(True)


class OptimReluMaxLinear(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mask):
        '''
        :param x: Matrix with shape (B, L, D) (Batch, sequence, embedding)
        :param weight: Matrix with shape (D, V) (embedding, vocabulary)
        :param bias: bias vector with shape (V)
        :return: the calcul of ReLU(Max_on_L(x @ w + b))
        '''

        output = x @ weight
        if bias is not None:
            output = output + bias.reshape(1, 1, bias.shape[0])

        mask = torch.where(mask == 1, 0, -torch.inf)
        output = output + mask.reshape(*mask.shape, 1)

        relu = ReLU()
        maximum, max_indice = torch.max(output, 1)
        result = relu(maximum)

        indice_not_zero = torch.nonzero(result, as_tuple=False)

        lmax = max_indice[indice_not_zero[:, 0], indice_not_zero[:, 1]]

        effective_indice = torch.cat((indice_not_zero, lmax.reshape(*lmax.shape, 1)), 1).cuda()

        ctx.save_for_backward(x, weight, bias, effective_indice)

        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, weight, bias, effective_indice = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = compute_gradient_x(effective_indice, grad_outputs[0].clone(), weight, torch.zeros_like(x))

        if ctx.needs_input_grad[1]:
            grad_weight = compute_gradient_w(effective_indice, grad_outputs[0].clone(), x, torch.zeros_like(weight))

        if ctx.needs_input_grad[2]:
            index = torch.zeros_like(grad_outputs[0], device='cuda', dtype=torch.int64)
            index[effective_indice[:, 0], effective_indice[:, 1]] = 1

            grad_bias = torch.zeros((2, grad_outputs[0].shape[1]), device='cuda',
                                    dtype=grad_outputs[0].dtype).scatter_add_(dim=0, index=index, src=grad_outputs[0])[
                        1, :]

        return grad_x, grad_weight, grad_bias, None

def combine_calculation(x, weight, bias, mask):
    output = x @ weight
    if bias is not None:
        output = output + bias.reshape(1, 1, bias.shape[0])

    mask = torch.where(mask == 1, 0, -torch.inf)
    output = output + mask.reshape(*mask.shape, 1)

    relu = ReLU()
    maximum, max_indice = torch.max(output, 1)
    result = relu(maximum)
    if (result.count_nonzero() / result.numel()) > 0.01:
        print(f'enter dense calculation \n')
        return result

    else:
        print(f'enter sparse calculation \n')
        return OptimReluMaxLinear.apply(x, weight, bias, mask)



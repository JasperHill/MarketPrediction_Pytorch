//////////////////////////////////////////////////////////////////////
//  RNN_cuda_passes.cpp
//  Feb. 2021 - J. Hill
//////////////////////////////////////////////////////////////////////
//
//  cuda-optimized counterpart to RNN_passes.cpp
//////////////////////////////////////////////////////////////////////
#include <torch/extension.h>
#include <torch/torch.h>
#include <stdlib.h>
#include <vector>


// forward and backward declarations

std::vector< torch::Tensor > LSTM_Op_cuda_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
					     torch::Tensor xOps, torch::Tensor hOps);

std::vector< torch::Tensor > LSTM_Op_cuda_backward(torch::Tensor x, torch::Tensor X, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
					      torch::Tensor xOps, torch::Tensor hOps, \
					      torch::Tensor grad_output);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector< torch::Tensor > LSTM_Op_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
					     torch::Tensor xOps, torch::Tensor hOps)
{
  CHECK_INPUT(x);
  CHECK_INPUT(c_p);
  CHECK_INPUT(h_p);
  CHECK_INPUT(xOps);
  CHECK_INPUT(hOps);

  return LSTM_Op_cuda_forward(x, c_p, h_p, xOps, hOps);
}


std::vector< torch::Tensor > LSTM_Op_backward(torch::Tensor x, torch::Tensor X, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
					       torch::Tensor xOps, torch::Tensor hOps, \
					       torch::Tensor grad_output)
{
  CHECK_INPUT(x);
  CHECK_INPUT(X);
  CHECK_INPUT(c);
  CHECK_INPUT(c_p);
  CHECK_INPUT(h_p);
  CHECK_INPUT(xOps);
  CHECK_INPUT(hOps);
  CHECK_INPUT(grad_output);

  return LSTM_Op_cuda_backward(x, X, c, c_p, h_p, xOps, hOps, grad_output);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("LSTM_Op_forward" , &LSTM_Op_forward, "LSTM_Op forward (CUDA)");
  m.def("LSTM_Op_backward", &LSTM_Op_backward, "LSTM_Op backward (CUDA)");
}

//////////////////////////////////////////////////////////////////////
//  RNN_passes.cpp
//  Feb. 2020 - J. Hill
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//  Implements LSTM_Op forward and backward passes
//  all functions are intended to operate on rank-2 tensors (batch, timestep)
//  input operators are passed as tensors of shape (4,output_dim,input_dim)
//  the size 4 corresponds to the input, forget, cell, and output operators
//////////////////////////////////////////////////////////////////////
#include <torch/extension.h>
#include <torch/torch.h>
#include <stdlib.h>
#include <vector>

//x      ==               input
//cs_p     ==         cell state (from previous iteration)
//hs_p   ==         hidden state
//iOp    ==      input operators
//fOp    ==     forget operators
//gOp    ==  cell gate operators
//oOp    ==     output operators

//h*_Op  ==      hidden operator

//currently, the forward pass is implemented under the assumption of cell and hidden states being of the same
//shape as the output vector
//
//additionally, the LSTM cell is configured to hold batch_size cell and hidden states so that it can learn
//to generate its states from each timestep of a single batch element rather than from each element of a batch

//in pytorch the sigmoid function is defined as sig(x) = 1/(1+exp(-x)) -> d/dx_sig(x) = sig(x)*(1 - sig(x))
torch::Tensor d_sig(torch::Tensor x)
{
  auto s = torch::sigmoid(x);
  return s*(1-s);
}

torch::Tensor d_tanh(torch::Tensor x)
{
  auto t = torch::tanh(x);
  return (1 - at::pow(t,2));
}

std::vector< at::Tensor > LSTM_Op_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
					  torch::Tensor xOps, torch::Tensor hOps)
{
  auto gate_xOps = xOps.chunk(4,0);//separates x operators into (input, forget, cell, output) operators
  auto gate_hOps = hOps.chunk(4,0);//ditto for hidden operators

  auto xi = torch::add(torch::mm(x, gate_xOps[0].transpose(0,1)), torch::mm(h_p, gate_hOps[0].transpose(0,1)));
  auto xf = torch::add(torch::mm(x, gate_xOps[1].transpose(0,1)), torch::mm(h_p, gate_hOps[1].transpose(0,1)));
  auto xg = torch::add(torch::mm(x, gate_xOps[2].transpose(0,1)), torch::mm(h_p, gate_hOps[2].transpose(0,1)));
  auto xo = torch::add(torch::mm(x, gate_xOps[3].transpose(0,1)), torch::mm(h_p, gate_hOps[3].transpose(0,1)));

  auto Xs = at::transpose(at::stack({xi,xf,xg,xo}), 0, 1);
  
  auto input_g = at::sigmoid(xi);
  auto forget_g = at::sigmoid(xf);
  auto cell_g = at::tanh(xg);
  auto output_g = at::sigmoid(xo); 
  
  auto c = forget_g*cell_g + input_g*cell_g;
  auto h = output_g*(at::tanh(c));

  return {Xs, c, h};
  
}//end of LSTM_Op_forward

std::vector< at::Tensor > LSTM_Op_backward(torch::Tensor x, torch::Tensor Xs, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
					   torch::Tensor xOps, torch::Tensor hOps, \
					   torch::Tensor grad_output)
{
  auto x_dims = x.sizes();
  auto y_dims = grad_output.sizes();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);

  int64_t n,i,j;
  int64_t batch_size = x_dims[0];
  int64_t input_dim = x_dims[1];
  int64_t output_dim = y_dims[-1];
 
  auto grad_input = torch::zeros_like(x, options);
  auto grad_xOps = torch::zeros({4, output_dim, input_dim}, options);
  auto grad_hOps = torch::zeros({4, output_dim, output_dim}, options);

  //todo: parallelize
  for (n = 0; n < batch_size; n++)
    {
      auto X = Xs[n];
      
      for (i = 0; i < output_dim; i++)
	{
	  auto dxo = grad_output[i] * d_sig(X[3])[i];
	  auto dh_dc = hOps[3][i] * at::sigmoid(X[3])[i] * d_tanh(c)[i];
	  
	  grad_xOps[n][3][i] = dxo * x[n][i];
	  grad_xOps[n][2][i] = dxo * dh_dc * at::sigmoid(X[1])[i] * d_tanh(X[2])[i] * x[n][i];
	  grad_xOps[n][1][i] = dxo * dh_dc * c_p[i] * d_sig(X[1])[i] * x[n][i];
	  grad_xOps[n][0][i] = dxo * dh_dc * at::tanh(X[2])[i] * d_sig(X[0])[i] * x[n][i];

	  grad_hOps[n][3][i] = dxo * h_p[n][i];
	  grad_hOps[n][2][i] = dxo * dh_dc * at::sigmoid(X[1])[i] * d_tanh(X[2]) * h_p[n][i];
	  grad_hOps[n][1][i] = dxo * dh_dc * c_p[i] * d_sig(X[1])[i] * h_p[n][i];
	  grad_hOps[n][0][i] = dxo * dh_dc * at::tanh(X[2])[i] * d_sig(X[0])[i] * h_p[n][i];

	  for (j = 0; j < input_dim; j++)
	    {
	      grad_input[n][j] += dxo * xOps[3][i][j];
	      grad_input[n][j] += dxo * dh_dc * at::sigmoid(X[1])[i][j] * d_tanh(X[2])[i][j] * xOps[n][2][i][j];
	      grad_input[n][j] += dxo * dh_dc * c_p[i] * d_sig(X[1])[i] * xOps[n][1][i][j];
	      grad_input[n][i] += dxo * dh_dc * at::tanh(X[2])[i] * d_sig(X[0])[i] * xOps[n][0][i][j];	      
	    }
	}
    }
  
  return {grad_input, grad_xOps, grad_hOps};
  
}//end of LSTM_Op_backward

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("LSTM_Op_forward" , &LSTM_Op_forward, "LSTM_Op forward");
  m.def("LSTM_Op_backward", &LSTM_Op_backward, "LSTM_Op backward");
}

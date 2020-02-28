//////////////////////////////////////////////////////////////////////
//  RNN_passes.cpp
//  Feb. 2020 - J. Hill
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//  Implements LSTM_Op forward and backward passes
//  all functions are intended to operate on rank-2 tensors (batch, timestep)
//  input operators are passed as tensors of shape (4,output_dim,input_dim)
//  x operators map inputs of length HIST_SIZE onto outputs of length TARG_SIZE == HIDDEN_SIZE
//  the size 4 corresponds to the input, forget, cell, and output operators
//////////////////////////////////////////////////////////////////////
#include <torch/extension.h>
#include <torch/torch.h>
#include <stdlib.h>
#include <vector>

//x    ==                  input
//cs_p  ==            cell state (from previous iteration)
//hs_p  ==          hidden state
//xOps  ==           x operators
//hOps  ==      hidden operators


//in pytorch the sigmoid function is defined as sig(x) = 1/(1+exp(-x)) -> d/dx_sig(x) = sig(x)*(1 - sig(x))
torch::Tensor d_sigmoid(torch::Tensor x)
{
  auto s = torch::sigmoid(x);
  return s*(1-s);
}

torch::Tensor d_tanh(torch::Tensor x)
{
  auto t = torch::tanh(x);
  return (1 - at::pow(t,2));
}

//x is of shape (batch_size, input_channels, timestep)
//xOps is of shape (gates, output_channels, input_channels, input_dim, output_dim)
//hOps is of shape (gates, output_channels, output_size, output_size)
std::vector< torch::Tensor > LSTM_Op_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
					  torch::Tensor xOps, torch::Tensor hOps)
{
  //auto gate_xOps = xOps.chunk(4,0);//separates x operators into (input, forget, cell, output) operators
  //auto gate_hOps = hOps.chunk(4,0);//ditto for hidden operators

  //std::cout << "x sizes: " << x.sizes() << std::endl;
  //std::cout << "op sizes: " << xOps[0].sizes() << std::endl;

  //std::cout << "h sizes: " << h_p.sizes() << std::endl;
  //std::cout << "hOp sizes: " << hOps[0].sizes() << std::endl;

  //std::cout<<"temp1 shape: "<<temp1.sizes()<<std::endl;
  //std::cout<<"temp2 shape: "<<temp2.sizes()<<std::endl;
  //contracting along the -1 axes of the vector and the operator corresponds to x^† * Op^†  
  auto xi = torch::add(at::tensordot(x, xOps[0], {-2, -1}, {-3, -1}), at::tensordot(h_p, hOps[0], {-2, -1}, {-3, -1}));
  auto xf = torch::add(at::tensordot(x, xOps[1], {-2, -1}, {-3, -1}), at::tensordot(h_p, hOps[1], {-2, -1}, {-3, -1}));
  auto xg = torch::add(at::tensordot(x, xOps[2], {-2, -1}, {-3, -1}), at::tensordot(h_p, hOps[2], {-2, -1}, {-3, -1}));
  auto xo = torch::add(at::tensordot(x, xOps[3], {-2, -1}, {-3, -1}), at::tensordot(h_p, hOps[3], {-2, -1}, {-3, -1}));
  
  auto X = at::stack({xi,xf,xg,xo});

  auto input_g = at::sigmoid(xi);
  auto forget_g = at::sigmoid(xf);
  auto cell_g = at::tanh(xg);
  auto output_g = at::sigmoid(xo); 
  
  auto c = forget_g*c_p + input_g*cell_g;
  auto h = output_g*(at::tanh(c));

  return {h, c, X};
  
}//end of LSTM_Op_forward

std::vector< torch::Tensor > LSTM_Op_backward(torch::Tensor x, torch::Tensor X, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
					      torch::Tensor xOps, torch::Tensor hOps, \
					      torch::Tensor grad_output)
{
  auto x_dims = x.sizes();
  auto y_dims = grad_output.sizes();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  

  int64_t n,i,ii,j,jj,k,kk;
  int64_t batch_size = x_dims[0];
  int64_t input_dim = x_dims[2];
  int64_t input_channels = x_dims[1];

  int64_t output_dim = y_dims[2];
  int64_t output_channels = y_dims[1];

  
  //transpose Xs to shape (batch_size, gates, output_channels, output_timesteps)
  auto X_t = at::transpose(X, 0, 1);
  auto grad_input = torch::zeros_like(x, options);
  auto grad_xOps = torch::zeros({4, output_channels, input_channels, output_dim, input_dim}, options);
  auto grad_hOps = torch::zeros({4, output_channels, input_channels, output_dim, output_dim}, options);

  //todo: parallelize
  for (n = 0; n < batch_size; n++)
    {
      auto c_n = c[n];
      auto h_p_n = h_p[n];
      auto x_n = x[n];
      auto X_n = X_t[n];
      
      auto grad_X_n = torch::zeros_like(X_n, options);  
      auto grad_output_n = grad_output[n];
      auto grad_input_n = grad_input[n];      

      for (i = 0; i < output_channels; i++)
	{
	  auto dh_dc = grad_output_n[i] * d_tanh(c_n[i]) * at::sigmoid(X_n[3][i]);
	  auto dh_dxo = grad_output_n[i] * at::tanh(c_n[i]) * d_sigmoid(X_n[3][i]);

	  grad_X_n[3] = grad_output_n[i] * at::tanh(c_n[i]) * d_sigmoid(X_n[3][i]);
	  grad_X_n[2] = dh_dc * at::sigmoid(X_n[0][i]) * d_tanh(X_n[2][i]);
	  grad_X_n[1] = dh_dc * c_p[n][i] * d_sigmoid(X_n[1][i]);//
	  grad_X_n[0] = dh_dc * at::tanh(X_n[2][i]) * d_sigmoid(X_n[0][i]);

	  for (j = 0; j < output_dim; j++)
	    {
	      for (ii = 0; ii < input_channels; ii++)
		{
		  for (k = 0; k < input_dim; k++)
		    {
		      grad_xOps[3][i][ii][j][k] += grad_X_n[3][i][j] * x_n[ii][k];
		      grad_xOps[2][i][ii][j][k] += grad_X_n[2][i][j] * x_n[ii][k];
		      grad_xOps[1][i][ii][j][k] += grad_X_n[1][i][j] * x_n[ii][k];
		      grad_xOps[0][i][ii][j][k] += grad_X_n[0][i][j] * x_n[ii][k];


		      //use input_channel indices for h_p_n because hOps generate outputs
		      //by acting on input channels of h_p
		      grad_hOps[3][i][ii][j][k] += grad_X_n[3][i][j] * h_p_n[ii][k];
		      grad_hOps[2][i][ii][j][k] += grad_X_n[2][i][j] * h_p_n[ii][k];
		      grad_hOps[1][i][ii][j][k] += grad_X_n[1][i][j] * h_p_n[ii][k];
		      grad_hOps[0][i][ii][j][k] += grad_X_n[0][i][j] * h_p_n[ii][k];
		      
		      grad_input_n[ii][k] += grad_X_n[3][i][j] * xOps[3][i][ii][j][k];
		      grad_input_n[ii][k] += grad_X_n[2][i][j] * xOps[2][i][ii][j][k];
		      grad_input_n[ii][k] += grad_X_n[1][i][j] * xOps[1][i][ii][j][k];
		      grad_input_n[ii][k] += grad_X_n[0][i][j] * xOps[0][i][ii][j][k];		      
		    }
		}

	      
	    }
	}
    }

  //c and h are not trainable parameters so return zero gradients
  auto grad_c_p = torch::zeros_like(c_p, options);
  auto grad_h_p = torch::zeros_like(h_p, options);
  
  return {grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps};
  
}//end of LSTM_Op_backward

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("LSTM_Op_forward" , &LSTM_Op_forward, "LSTM_Op forward");
  m.def("LSTM_Op_backward", &LSTM_Op_backward, "LSTM_Op backward");
}

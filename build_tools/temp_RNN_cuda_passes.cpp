//////////////////////////////////////////////////////////////////////
//  RNN_cuda_passes.cpp
//  Feb. 2021 - J. Hill
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//  Implements LSTM_Op forward and backward passes for cuda-enabled GPUs
//  all functions are intended to operate on rank-3 tensors (batch, high/low, timestep)
//  input operators are passed as tensors of shape (4,output_channels, output_dim, input_channels, input_dim)
//  x operators map inputs of length HIST_SIZE onto outputs of length TARG_SIZE == HIDDEN_SIZE
//  the size 4 corresponds to the input, forget, cell, and output operators
//////////////////////////////////////////////////////////////////////
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>

//in pytorch the sigmoid function is defined as sig(x) = 1/(1+exp(-x)) -> d/dx_sig(x) = sig(x)*(1 - sig(x))
template <typename scalar_t>
__device__ __forceinline__ d_sigmoid(scalar_t z)
{
  const auto t = sigmoid(z);
  return t*(1-t);
}

template <typename scalar_t>
__device__ __forceinline__ d_tanh(scalar_t z)
{
  auto t = tanh(z);
  return (1 - (t*t));
}

//x is of shape (batch_size, input_channels, timestep)
//xOps is of shape (gates, output_channels, input_channels, output_dim, input_dim)
//hOps is of shape (gates, output_channels, output_size, output_size)

std::vector< torch::Tensor > LSTM_Op_cuda_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
					 			 torch::Tensor xOps, torch::Tensor hOps)
{
  const auto x_dims = x.sizes();
  const auto batch_size = x_dims[0];
  const auto state_size = x_dims[1]*x_dims[2];
  
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1)/threads, batch_size);
  
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

  return {c, h, X};
  
}//end of LSTM_Op_forward


//x is of shape (batch_size, currency, input_channels, timestep)
//xOps is of shape (gates, output_currency, input_currency, output_channels, input_channels, output_dim, input_dim)
//hOps is of shape (gates, output_currency, input_currency, output_channels, output_channels output_dim, output_dim)
std::vector< torch::Tensor > MC_LSTM_Op_forward(torch::Tensor x, torch::Tensor c_p, torch::Tensor h_p, \
						torch::Tensor xOps, torch::Tensor hOps)
{
  //contracting along the -1 axes of the vector and the operator corresponds to x^† * Op^†
  auto xi = torch::add(at::tensordot(x, xOps[0], {-3, -2, -1}, {-5, -3, -1}), at::tensordot(h_p, hOps[0], {-3, -2, -1}, {-5, -3, -1}));
  auto xf = torch::add(at::tensordot(x, xOps[1], {-3, -2, -1}, {-5, -3, -1}), at::tensordot(h_p, hOps[1], {-3, -2, -1}, {-5, -3, -1}));
  auto xg = torch::add(at::tensordot(x, xOps[2], {-3, -2, -1}, {-5, -3, -1}), at::tensordot(h_p, hOps[2], {-3, -2, -1}, {-5, -3, -1}));
  auto xo = torch::add(at::tensordot(x, xOps[3], {-3, -2, -1}, {-5, -3, -1}), at::tensordot(h_p, hOps[3], {-3, -2, -1}, {-5, -3, -1}));

  auto X = at::stack({xi,xf,xg,xo});

  auto input_g = at::sigmoid(xi);
  auto forget_g = at::sigmoid(xf);
  auto cell_g = at::tanh(xg);
  auto output_g = at::sigmoid(xo); 
  
  auto c = forget_g*c_p + input_g*cell_g;
  auto h = output_g*(at::tanh(c));

  return {c, h, X};
  
}//end of LSTM_Op_forward


std::vector< torch::Tensor > LSTM_Op_backward(torch::Tensor x, torch::Tensor X, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
					      torch::Tensor xOps, torch::Tensor hOps, \
					      torch::Tensor grad_output)
{
  //x is of shape (batch_size, gates, input_channels, input_timesteps)
  auto x_dims = x.sizes();
  auto y_dims = grad_output.sizes();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  

  int64_t n,i,ii,j,jj,k;
  int64_t batch_size = x_dims[0];
  int64_t input_dim = x_dims[2];
  int64_t input_channels = x_dims[1];

  int64_t output_dim = y_dims[2];
  int64_t output_channels = y_dims[1];

  
  //transpose Xs to shape (batch_size, gates, output_channels, output_timesteps)
  auto X_t = at::transpose(X, 0, 1);
  auto grad_input = torch::zeros_like(x, options);
  auto grad_xOps = torch::zeros({4, output_channels, input_channels, output_dim, input_dim}, options);
  auto grad_hOps = torch::zeros({4, output_channels, output_channels, output_dim, output_dim}, options);

  //todo: parallelize
#pragma omp parallel default(none) schedule(dynamic)			\
  firstprivate(x, X_t, c, c_p, h_p, batch_size, output_channels, output_dim, input_channels, input_dim) \
  shared(grad_input, grad_xOps, grad_hOps)				\
  private(n,i,ii,j,jj,k)
  {
    #pragma omp for nowait
    
  for (n = 0; n < batch_size; n++)
    {
      auto c_n = c[n];
      auto h_p_n = h_p[n];
      auto x_n = x[n];
      auto X_n = X_t[n];
      
      auto grad_X_n = torch::zeros_like(X_n, options);  
      auto grad_output_n = grad_output[n];
      auto grad_input_n = grad_input[n];      

      auto dh_dc = grad_output_n * d_tanh(c_n) * at::sigmoid(X_n[3]);
      auto dh_dxo = grad_output_n * at::tanh(c_n) * d_sigmoid(X_n[3]);
      
      grad_X_n[3] = grad_output_n * at::tanh(c_n) * d_sigmoid(X_n[3]);
      grad_X_n[2] = dh_dc * at::sigmoid(X_n[0]) * d_tanh(X_n[2]);
      grad_X_n[1] = dh_dc * c_p[n] * d_sigmoid(X_n[1]);//
      grad_X_n[0] = dh_dc * at::tanh(X_n[2]) * d_sigmoid(X_n[0]);

      
      for (i = 0; i < output_channels; i++)
	{
	  /*auto dh_dc = grad_output_n[i] * d_tanh(c_n[i]) * at::sigmoid(X_n[3][i]);
	  auto dh_dxo = grad_output_n[i] * at::tanh(c_n[i]) * d_sigmoid(X_n[3][i]);

	  grad_X_n[3] = grad_output_n[i] * at::tanh(c_n[i]) * d_sigmoid(X_n[3][i]);
	  grad_X_n[2] = dh_dc * at::sigmoid(X_n[0][i]) * d_tanh(X_n[2][i]);
	  grad_X_n[1] = dh_dc * c_p[n][i] * d_sigmoid(X_n[1][i]);//
	  grad_X_n[0] = dh_dc * at::tanh(X_n[2][i]) * d_sigmoid(X_n[0][i]);*/

	  for (j = 0; j < output_dim; j++)
	    {
	      for (ii = 0; ii < input_channels; ii++)
		{
		  for (jj = 0; jj < input_dim; jj++)
		    {
		      grad_xOps[3][i][ii][j][jj] += grad_X_n[3][i][j] * x_n[ii][jj];
		      grad_xOps[2][i][ii][j][jj] += grad_X_n[2][i][j] * x_n[ii][jj];
		      grad_xOps[1][i][ii][j][jj] += grad_X_n[1][i][j] * x_n[ii][jj];
		      grad_xOps[0][i][ii][j][jj] += grad_X_n[0][i][j] * x_n[ii][jj];
		      
		      grad_input_n[ii][jj] += grad_X_n[3][i][j] * xOps[3][i][ii][j][jj];
		      grad_input_n[ii][jj] += grad_X_n[2][i][j] * xOps[2][i][ii][j][jj];
		      grad_input_n[ii][jj] += grad_X_n[1][i][j] * xOps[1][i][ii][j][jj];
		      grad_input_n[ii][jj] += grad_X_n[0][i][j] * xOps[0][i][ii][j][jj];		      
		    }//jj
		}//ii
	      
	      for (ii = 0; ii < output_channels; ii++)
		{
		  for (jj = 0; jj < output_dim; jj++)
		    {
		      //use input_channel indices for h_p_n because hOps generate outputs
		      //by acting on input channels of h_p		      
		      grad_hOps[3][i][ii][j][jj] += grad_X_n[3][i][j] * h_p_n[ii][jj];
		      grad_hOps[2][i][ii][j][jj] += grad_X_n[2][i][j] * h_p_n[ii][jj];
		      grad_hOps[1][i][ii][j][jj] += grad_X_n[1][i][j] * h_p_n[ii][jj];
		      grad_hOps[0][i][ii][j][jj] += grad_X_n[0][i][j] * h_p_n[ii][jj];		      		      
		    }//jj
		}//ii
	    }//j
	}//i
    }//n
  }
  //c and h are not trainable parameters so return zero gradients
  auto grad_c_p = torch::zeros_like(c_p, options);
  auto grad_h_p = torch::zeros_like(h_p, options);
  
  return {grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps};
  
}//end of LSTM_Op_backward


///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


std::vector< torch::Tensor > MC_LSTM_Op_backward(torch::Tensor x, torch::Tensor X, torch::Tensor c, torch::Tensor c_p, torch::Tensor h_p, \
						 torch::Tensor xOps, torch::Tensor hOps, \
						 torch::Tensor grad_output)
{
  //x is of shape (batch_size, input_currencies, input_channels, input_dim)
  auto x_dims = x.sizes();
  auto y_dims = grad_output.sizes();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  

  int64_t n,nco,nnco,nci,i,ii,j,jj;
  int64_t batch_size = x_dims[0];
  int64_t input_currencies = x_dims[1];
  int64_t input_dim = x_dims[3];
  int64_t input_channels = x_dims[2];

  int64_t output_currencies = y_dims[1];
  int64_t output_dim = y_dims[3];
  int64_t output_channels = y_dims[2];

  //transpose Xs to shape (batch_size, gates, currencies, output_channels, output_timesteps)
  auto X_t = at::transpose(X, 0, 1);
  auto grad_input = torch::zeros_like(x, options);
  auto grad_xOps = torch::zeros({4, output_currencies, input_currencies, output_channels, input_channels, output_dim, input_dim}, options);
  auto grad_hOps = torch::zeros({4, output_currencies, output_currencies, output_channels, output_channels, output_dim, output_dim}, options);

  //todo: parallelize
#pragma omp parallel default(none) schedule(dynamic)			\
  firstprivate(x, X_t, c, c_p, h_p, batch_size, output_channels, output_dim, input_channels, input_dim) \
  shared(grad_input, grad_xOps, grad_hOps)				\
  private(n,nco,nnco,nci,i,ii,j,jj)
  {
    #pragma omp for nowait
    
  for (n = 0; n < batch_size; n++)
    {
      auto c_n = c[n];
      auto h_p_n = h_p[n];
      auto x_n = x[n];
      auto X_n = X_t[n];
      
      auto grad_X_n = torch::zeros_like(X_n, options);  
      auto grad_output_n = grad_output[n];
      auto grad_input_n = grad_input[n];      

      auto dh_dc = grad_output_n * d_tanh(c_n) * at::sigmoid(X_n[3]);
      grad_X_n[3] = grad_output_n * at::tanh(c_n) * d_sigmoid(X_n[3]);
      grad_X_n[2] = dh_dc * at::sigmoid(X_n[0]) * d_tanh(X_n[2]);
      grad_X_n[1] = dh_dc * c_p[n] * d_sigmoid(X_n[1]);
      grad_X_n[0] = dh_dc * at::tanh(X_n[2]) * d_sigmoid(X_n[0]);

      for (nco = 0; nco < output_currencies; nco++)
	{
	  for (i = 0; i < output_channels; i++)
	    {
	      for (nci = 0; nci < input_currencies; nci++)
		{
		  for (j = 0; j < output_dim; j++)
		    {
		      for (ii = 0; ii < input_channels; ii++)
			{
			  for (jj = 0; jj < input_dim; jj++)
			    {
			      grad_xOps[3][nco][nci][i][ii][j][jj] += grad_X_n[3][nco][i][j] * x_n[nci][ii][jj];
			      grad_xOps[2][nco][nci][i][ii][j][jj] += grad_X_n[2][nco][i][j] * x_n[nci][ii][jj];
			      grad_xOps[1][nco][nci][i][ii][j][jj] += grad_X_n[1][nco][i][j] * x_n[nci][ii][jj];
			      grad_xOps[0][nco][nci][i][ii][j][jj] += grad_X_n[0][nco][i][j] * x_n[nci][ii][jj];

			      grad_input_n[nci][ii][jj] += grad_X_n[3][nco][i][j] * xOps[3][nco][nci][i][ii][j][jj];
			      grad_input_n[nci][ii][jj] += grad_X_n[2][nco][i][j] * xOps[2][nco][nci][i][ii][j][jj];
			      grad_input_n[nci][ii][jj] += grad_X_n[1][nco][i][j] * xOps[1][nco][nci][i][ii][j][jj];
			      grad_input_n[nci][ii][jj] += grad_X_n[0][nco][i][j] * xOps[0][nco][nci][i][ii][j][jj];
			    }//jj
			}//ii  
		    }//j
		}//nci
	      
	      for (nnco = 0; nnco < output_currencies; nnco++)
		{
		  for (j = 0; j < output_dim; j++)
		    {
		      for (ii = 0; ii < output_channels; ii++)
			{
			  for (jj = 0; jj < output_dim; jj++)
			    {
			      //use input_channel indices for h_p_n because hOps generate outputs
			      //by acting on input channels of h_p		      
			      grad_hOps[3][nco][nci][i][ii][j][jj] += grad_X_n[3][nco][i][j] * h_p_n[nco][ii][jj];
			      grad_hOps[2][nco][nci][i][ii][j][jj] += grad_X_n[2][nco][i][j] * h_p_n[nco][ii][jj];
			      grad_hOps[1][nco][nci][i][ii][j][jj] += grad_X_n[1][nco][i][j] * h_p_n[nco][ii][jj];
			      grad_hOps[0][nco][nci][i][ii][j][jj] += grad_X_n[0][nco][i][j] * h_p_n[nco][ii][jj];
			    }//jj
			}//ii
		    }//j
		}//nnco
	    }//i
	}//nco
    }//n
  }
  //c and h are not trainable parameters so return zero gradients
  auto grad_c_p = torch::zeros_like(c_p, options);
  auto grad_h_p = torch::zeros_like(h_p, options);
  
  return {grad_input, grad_c_p, grad_h_p, grad_xOps, grad_hOps};
  
}//end of MC_LSTM_Op_backward



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("LSTM_Op_forward" , &LSTM_Op_forward, "LSTM_Op forward");
  m.def("LSTM_Op_backward", &LSTM_Op_backward, "LSTM_Op backward");
  
  m.def("MC_LSTM_Op_forward", &MC_LSTM_Op_forward, "MC_LSTM_Op forward");
  m.def("MC_LSTM_Op_backward", &MC_LSTM_Op_backward, "MC_LSTM_Op backward");
}

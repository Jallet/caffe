#include <vector>
#include <string>

#include "caffe/layers/euclidean_loss_with_tv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void GPUSign(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index] > 0 ? 1 : -1;
    }
}

template <typename Dtype>
void EuclideanLossWithTVLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  //Dtype dot = caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data());
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //if (height <=1 && width <= 1) {
  //  top[0]->mutable_gpu_data()[0] = loss;
  //  return;
  //}
  Dtype row_tv = 0;
  Dtype col_tv = 0;
  
  LOG_IF(INFO, Caffe::root_solver())
      << "original loss: " << loss;
  //print_data(num_, "bottom[0]", bottom[0]->gpu_data(), height, width);
  //print_data(num_, "bottom[1]", bottom[1]->gpu_data(), height, width);
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  if (height > 1) {
    Dtype* row_data = row_.mutable_gpu_data();
    const Dtype* q_data = _2d_q_.gpu_data();
    LOG_IF(INFO, Caffe::root_solver())
        << "num_ : " << num_;
    for (int i = 0; i < num_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height - 1, width, height, 
              (Dtype)1., q_data, bottom_data + height * width * i, 
              (Dtype)0., row_data + i * (height - 1) * width);
    }
    //for (int i = 0; i < row_.count(); ++i) {
    //    Dtype square = (Dtype)0.5 *  *(row_data + i) * *(row_data + i);
    //    if (square < 1) {
    //        caffe_gpu_set(1, square, row_data + i);
    //    }
    //}
    //LOG_IF(INFO, Caffe::root_solver())
    //    << "q * row finished";

    //print_data(num_, "row_", row_.gpu_data(), height - 1, width);

    Dtype *abs_row_data = abs_row_.mutable_gpu_data();
    caffe_gpu_abs(num_ * (height - 1) * width , row_data, abs_row_data); 
    //row_tv = caffe_gpu_asum(num_ * (height - 1) * width, abs_row_data);
    caffe_gpu_asum(num_ * (height - 1) * width, abs_row_data, &row_tv);
    //print_data(num_, "abs_row_", abs_row_.gpu_data(), height - 1, width);
  }

  if (width > 1) {
    Dtype* col_data = col_.mutable_gpu_data();
    const Dtype* p_data = _2d_p_.gpu_data();
    for (int i = 0; i < num_; ++i) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width - 1, width, 
              (Dtype)1., bottom_data + height * width * i, p_data, (Dtype)0., 
              col_data + height * (width - 1) * i); 
    }
    //for (int i = 0; i < col_.count(); ++i) {
    //    Dtype square = (Dtype)0.5 * *(col_data + i) * *(col_data + i);
    //    if (square < 1) {
    //        caffe_gpu_set(1, square, col_data + i);
    //    }
    //}
    //print_data(num_, "col_", col_.gpu_data(), height, width - 1);
    Dtype *abs_col_data = abs_col_.mutable_gpu_data();
    caffe_gpu_abs(num_ * height * (width - 1), col_data, abs_col_data); 
    //col_tv = caffe_gpu_asum(num_ * height * (width - 1), abs_col_data);
    caffe_gpu_asum(num_ * height * (width - 1), abs_col_data, &col_tv);
    //print_data(num_, "abs_col_", abs_col_.gpu_data(), height, width - 1);
  }

  Dtype tv_loss = (row_tv + col_tv) / bottom[0]->num();

  loss = loss + tv_loss;
  top[0]->mutable_cpu_data()[0] = loss;

  LOG_IF(INFO, Caffe::root_solver())
      << "top[0]->diff()[0]: " << top[0]->cpu_diff()[0];

  LOG_IF(INFO, Caffe::root_solver())
      << "row_tv: " << row_tv << "col_tv: " << col_tv;
  //loss = loss + col_tv + row_tv;
  LOG_IF(INFO, Caffe::root_solver())
      << "smoothed loss: " << loss;
}

template <typename Dtype>
void EuclideanLossWithTVLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Dtype* total_gradient_data = total_gradient.mutable_gpu_data();
  Dtype* row_total_gradient_data = row_total_gradient.mutable_gpu_data();
  Dtype* col_total_gradient_data = col_total_gradient.mutable_gpu_data();
  caffe_gpu_set(total_gradient.count(), (Dtype)0., total_gradient_data);
  caffe_gpu_set(row_total_gradient.count(), (Dtype)0., row_total_gradient_data);
  caffe_gpu_set(col_total_gradient.count(), (Dtype)0., col_total_gradient_data);
  const Dtype* row_gradient_adder_data = row_gradient_adder.gpu_data();
  const Dtype* col_gradient_adder_data = col_gradient_adder.gpu_data();

  LOG_IF(INFO, Caffe::root_solver())
      << "Backwarding";
  if (height > 1) {
    const Dtype* row_data = row_.gpu_data();
    Dtype* row_sign_data = row_sign_.mutable_gpu_data();
    const Dtype* row_gradient_comp_data = row_gradient_completer.gpu_data();
    Dtype* row_gradient_data = row_gradient_.mutable_gpu_data();
  LOG_IF(INFO, Caffe::root_solver())
      << "In height > 1";
  const int count = row_.count();
  GPUSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          row_.count(), row_data, row_sign_data);
  CUDA_POST_KERNEL_CHECK;
  LOG_IF(INFO, Caffe::root_solver())
      << "GPUSign row finish";
    //for (int i = 0; i < row_.count(); ++i) {
    //    LOG_IF(INFO, Caffe::root_solver())
    //        << "i: " << i;
    //    //Dtype x = row_data[i];
    //    LOG_IF(INFO, Caffe::root_solver())
    //        << "x:";
    //    //LOG_IF(INFO, Caffe::root_solver())
    //    //    << "x: " << x;
    //    if (*(row_data + i) > 0) {
    //      LOG_IF(INFO, Caffe::root_solver())
    //          << "row_data > 0";
    //      caffe_gpu_set(1, (Dtype)1., row_sign_data + i);
    //      LOG_IF(INFO, Caffe::root_solver())
    //          << "set row_data 1";
    //    } else {
    //      LOG_IF(INFO, Caffe::root_solver())
    //          << "row_data < 0";
    //      caffe_gpu_set(1, (Dtype)-1., row_sign_data + i);
    //      LOG_IF(INFO, Caffe::root_solver())
    //          << "set row_data -1";
    //    }
    //    //if (*(row_data + i) > 1) {
    //    //  caffe_gpu_set(1, (Dtype)1., row_sign_data + i);
    //    //} else if (*(row_data + i) < -1) {
    //    //  caffe_gpu_set(1, (Dtype)-1., row_sign_data + i);
    //    //} else {
    //    //    const Dtype square = (Dtype)2. * *(row_data + i);
    //    //    caffe_sqr(1, &square, row_sign_data + i);
    //    //    //caffe_gpu_set(1, (Dtype)(*(row_data + i)), row_sign_data + i);
    //    //}
    //}
  LOG_IF(INFO, Caffe::root_solver())
      << "after setting row_data";
    //print_data(num_, "row_", row_.gpu_data(), height - 1, width);
    //print_data(num_, "row_sign", row_sign_.gpu_data(), height - 1, width);
    //print_data(1, "row_gradient_adder", row_gradient_adder.gpu_data(), height, height);
    //print_data(1, "row_comp_data", row_gradient_comp_data, height, height - 1);
    
    //for (int i = 0; i < num_; ++i) {
    //  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
    //          (Dtype)1., row_data + height * (width - 1) * i, row_gradient_comp_data, (Dtype)0., 
    //          row_gradient_data + height * width * i); 
    //  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
    //          (Dtype)1., row_gradient_data + height * width * i, row_gradient_adder_data, (Dtype)0., 
    //          row_total_gradient_data + height * width * i); 
    //}
    for (int i = 0; i < num_; ++i) {
      //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
      //        (Dtype)1., row_data + height * (width - 1) * i, row_gradient_comp_data, (Dtype)0., 
      //        row_gradient_data + height * width * i); 
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height - 1, 
              (Dtype)1., row_gradient_comp_data, row_sign_data + (height - 1) * width * i, (Dtype)0., 
              row_gradient_data + height * width * i); 
      //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
      //        (Dtype)1., row_gradient_data + height * width * i, row_gradient_adder_data, (Dtype)0., 
      //        row_total_gradient_data + height * width * i); 
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height, 
              (Dtype)1., row_gradient_adder_data, row_gradient_data + height * width * i, (Dtype)0., 
              row_total_gradient_data + height * width * i); 
    }
    
    LOG_IF(INFO, Caffe::root_solver())
        << "after calculating row_gradient";
    //print_data(num_, "row_gradient", row_gradient_.gpu_data(), height, width);
    //print_data(num_, "row_total_gradient", row_total_gradient.gpu_data(), height, width);
  }
  if (width > 1) {
    const Dtype* col_data = col_.gpu_data();
    Dtype* col_sign_data = col_sign_.mutable_gpu_data();
    const Dtype* col_gradient_comp_data = col_gradient_completer.gpu_data();
    Dtype* col_gradient_data = col_gradient_.mutable_gpu_data();
    int count = col_.count();
    GPUSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, col_data, col_sign_data);
    CUDA_POST_KERNEL_CHECK;
    //for (int i = 0; i < col_.count(); ++i) {
    //    if (*(col_data + i) > 0) {
    //      caffe_gpu_set(1, (Dtype)1., col_sign_data + i);
    //    } else {
    //      caffe_gpu_set(1, (Dtype)-1., col_sign_data + i);
    //    }
    //    //if (*(col_data + i) > 1) {
    //    //  caffe_gpu_set(1, (Dtype)1., col_sign_data + i);
    //    //} else if (*(col_data + i) < -1) {
    //    //  caffe_gpu_set(1, (Dtype)-1., col_sign_data + i);
    //    //} else {
    //    //    const Dtype square = (Dtype)2. * *(col_data + i);
    //    //    caffe_sqr(1, &square, col_sign_data + i);
    //    //    //caffe_gpu_set(1, (Dtype)(*(col_data + i)), col_sign_data + i);
    //    //}
    //}
    //print_data(num_, "col_", col_.gpu_data(), height, width - 1);
    //print_data(num_, "col_sign", col_sign_.gpu_data(), height, width - 1);
    //print_data(1, "col_gradient_adder", col_gradient_adder.gpu_data(), width, width);
    //print_data(1, "col_comp_data", col_gradient_comp_data, width - 1, width);
    for (int i = 0; i < num_; ++i) {
      //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height - 1, 
      //        (Dtype)1., col_gradient_comp_data, col_data + (height -1) * width * i, (Dtype)0., 
      //        col_gradient_data + height * width * i); 
      //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
      //        (Dtype)1., col_gradient_adder_data, col_gradient_data + height * width * i, (Dtype)0., 
      //        col_total_gradient_data + height * width * i); 
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
              (Dtype)1., col_sign_data + height * (width - 1) * i, col_gradient_comp_data, (Dtype)0., 
              col_gradient_data + height * width * i); 
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
              (Dtype)1., col_gradient_data + height * width * i, col_gradient_adder_data, (Dtype)0., 
              col_total_gradient_data + height * width * i); 
    }
    //print_data(num_, "col_gradient", col_gradient_.gpu_data(), height, width);
    //print_data(num_, "col_total_gradient", col_total_gradient.gpu_data(), height, width);
  }

  caffe_gpu_add(total_gradient.count(), row_total_gradient_data, 
          col_total_gradient_data, total_gradient_data);
  //print_data(num_, "total_gradient_data", total_gradient_data,  height, width);
  


  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      LOG_IF(INFO, Caffe::root_solver())
          << "top.shape: " << top[0]->shape_string();
      LOG_IF(INFO, Caffe::root_solver())
          << "bottom.shape: " << bottom[0]->shape_string();
      LOG_IF(INFO, Caffe::root_solver())
          << "bottom.num: " << bottom[i]->num();
      LOG_IF(INFO, Caffe::root_solver())
          << "bottom.count: " << bottom[i]->count();
      LOG_IF(INFO, Caffe::root_solver())
          << "i: " << i << ", top[0]->cpu_diff[0]: " << top[0]->cpu_diff()[0];
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      LOG_IF(INFO, Caffe::root_solver())
          << "alpha: " << alpha;
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
      //print_data(num_, "bottom-diff", bottom[i]->gpu_diff(), height, width);
      //const Dtype a = top[0]->cpu_diff()[0] / bottom[i]->num();
      const Dtype a = top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_scal(total_gradient.count(), a, total_gradient_data);
      //if (sign > 0) {
      //    const Dtype a = alpha; 
      //    caffe_scal(total_gradient.count(), a, total_gradient_data);
      //} else {
      //    const Dtype a = -1 * alpha;
      //    caffe_scal(total_gradient.count(), a, total_gradient_data);
      //}
      //const Dtype a= sign / bottom[i]->num();
      //print_data(num_, "total_gradient_data", total_gradient.gpu_data(), height, width);
      //caffe_gpu_axpby(
      //    total_gradient.count(),              // count
      //    alpha,                              // alpha
      //    total_gradient.gpu_data(),                   // a
      //    Dtype(0),                           // beta
      //    total_gradient_data);  // b
      //caffe_gpu_axpby(
      //    total_gradient.count(),              // count
      //    alpha,                              // alpha
      //    total_gradient.gpu_data(),                   // a
      //    Dtype(1),                           // beta
      //    bottom[i]->mutable_cpu_diff());  // b
      //print_data(1, "total_gradient_data", total_gradient.gpu_data(), height, width);
      //print_data(1, "total_gradient_data", total_gradient.gpu_data(), height, width);
      caffe_gpu_axpy(total_gradient.count(), (Dtype)1, 
              total_gradient_data, bottom[i]->mutable_gpu_diff());
      //caffe_gpu_add(total_gradient.count(), total_gradient_data, 
      //        bottom[i]->cpu_diff(), bottom[i]->mutable_cpu_diff());
      //print_data(num_, i + "bottom-diff", bottom[i]->gpu_diff(), height, width);
      LOG_IF(INFO, Caffe::root_solver())
          << "add TV gradient";
    }
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossWithTVLayer);

}  // namespace caffe

#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  Dtype* diff_data = diff_.mutable_cpu_data();
  temp_.ReshapeLike(*bottom[0]);
  Dtype* temp_data = temp_.mutable_cpu_data();
  caffe_set(temp_.count(), (Dtype)1., temp_data);
  p_.ReshapeLike(*bottom[0]);
  Dtype* p_data = p_.mutable_cpu_data();
  caffe_set(p_.count(), (Dtype)1., p_data);
  q_.ReshapeLike(*bottom[0]);
  Dtype* q_data = q_.mutable_cpu_data();
  caffe_set(q_.count(), (Dtype)1., q_data);
  r_.ReshapeLike(*bottom[0]);
  Dtype* r_data = r_.mutable_cpu_data();
  caffe_set(r_.count(), (Dtype)1., r_data);
  vector<int> shape = vector<int>(1, 2);
  s_.Reshape(shape);
  Dtype* s_data = s_.mutable_cpu_data();
  t_.Reshape(shape);
  Dtype* t_data = t_.mutable_cpu_data();
  u_.Reshape(shape);
  Dtype* u_data = u_.mutable_cpu_data();
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "bottom[i].shape" << bottom[i]->shape_string();
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe

#include <vector>
#include <string>

#include "caffe/layers/euclidean_loss_with_tv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
//void EuclideanLossWithTVLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
//        const vector<Blob<Dtype>*>& top) {
//  //EuclideanWithTVParameter param = this->layer_param_.euclidean_with_tv_param();
//  //lambda_ = param.lambda();
//}

template <typename Dtype>
void EuclideanLossWithTVLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  vector<int> bottom_shape = bottom[0]->shape();
  num_ = bottom[0]->count() / bottom_shape[bottom_shape.size() - 1]
      / bottom_shape[bottom_shape.size() - 2];
  height = bottom_shape[bottom_shape.size() - 2];
  width = bottom_shape[bottom_shape.size() - 1];

  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "capacity: " << bottom_shape.capacity() << "count: " << bottom_shape.size();

  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "height: " << height << ", width: " << width;
  
  vector<int> row_adder_shape;
  row_adder_shape.push_back(height);
  row_adder_shape.push_back(height);
  
  vector<int> col_adder_shape;
  col_adder_shape.push_back(width);
  col_adder_shape.push_back(width);


  row_gradient_adder.Reshape(row_adder_shape);
  col_gradient_adder.Reshape(col_adder_shape);
  
  total_gradient.Reshape(bottom_shape);
  row_total_gradient.Reshape(bottom_shape);
  col_total_gradient.Reshape(bottom_shape);
  
  Dtype* row_gradient_adder_data = row_gradient_adder.mutable_cpu_data();
  caffe_set(row_gradient_adder.count(), (Dtype)0., row_gradient_adder_data);
  Dtype* col_gradient_adder_data = col_gradient_adder.mutable_cpu_data();
  caffe_set(col_gradient_adder.count(), (Dtype)0., col_gradient_adder_data);
  
  for (int i = 1; i < row_adder_shape[0]; ++i) {
    caffe_set(1, (Dtype)1., row_gradient_adder_data + (i - 1) * row_adder_shape[1] + i - 1);
    caffe_set(1, (Dtype)-1., row_gradient_adder_data + i * row_adder_shape[1] + i - 1 );
  }
  caffe_set(1, (Dtype)1., row_gradient_adder_data + row_adder_shape[0] * row_adder_shape[1] - 1); 
  ////print_data(1, "row_gradient_adder", 
  //        row_gradient_adder_data, row_adder_shape[0], col_adder_shape[1]);
  //for (int i = 0; i < row_adder_shape[0]; ++i) {
  //    for (int j = 0; j < row_adder_shape[1]; ++j) {
  //        if (j + 1 == i) {
  //            caffe_set(1, (Dtype)1., row_gradient_adder_data + i * row_adder_shape[1] + j);
  //        }
  //        if (i + 1 == j) {
  //            caffe_set(1, (Dtype)-1., row_gradient_adder_data + i * row_adder_shape[1] + j);
  //        }
  //    }
  //}
  
  //for (int i = 0; i < col_adder_shape[0]; ++i) {
  //    for (int j = 0; j < col_adder_shape[1]; ++j) {
  //        if (j + 1 == i) {
  //            caffe_set(1, (Dtype)-1., col_gradient_adder_data + i * col_adder_shape[1] + j);
  //        }
  //        if (i + 1 == j) {
  //            caffe_set(1, (Dtype)1., col_gradient_adder_data + i * col_adder_shape[1] + j);
  //        }
  //    }
  //}

  for (int i = 1; i < col_adder_shape[0]; ++i) {
    caffe_set(1, (Dtype)1., col_gradient_adder_data + (i - 1) * col_adder_shape[1] + i - 1);
    caffe_set(1, (Dtype)-1., col_gradient_adder_data + i * col_adder_shape[1] + i - 1);
  }
  caffe_set(1, (Dtype)1., col_gradient_adder_data + col_adder_shape[0] * col_adder_shape[1] - 1); 
  ////print_data(1, "col_gradient_adder", 
   //       col_gradient_adder_data, col_adder_shape[0], col_adder_shape[1]);

  total_gradient.Reshape(bottom_shape);
  
  if (height > 1) {
    row_gradient_.Reshape(bottom_shape);
    //row_total_gradient.Reshape(bottom_shape);
    vector<int> row_comp_shape;
    row_comp_shape.push_back(height);
    row_comp_shape.push_back(height - 1);
    row_gradient_completer.Reshape(row_comp_shape);
    //row_gradient_completer.Reshape(bottom_shape);
    Dtype* row_gradient_data = row_gradient_completer.mutable_cpu_data();
    caffe_set(row_gradient_completer.count(), (Dtype)0., row_gradient_data);
    
    for (int j = 0; j < row_comp_shape[1]; ++j) {
        caffe_set(1, (Dtype)1., row_gradient_data + j * row_comp_shape[1] + j);
    }
    vector<int> q_shape = bottom_shape;
    q_shape[q_shape.size() - 1] = q_shape[q_shape.size() - 2];
    q_shape[q_shape.size() - 2] = q_shape[q_shape.size() - 2] -1;
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "q_shape: " << q_shape[0] << ", " 
    //    << q_shape[1] << ", " << q_shape[2] << ", "
    //    << q_shape[3];
    vector<int> row_shape = bottom_shape;
    row_shape[row_shape.size() - 2] -= 1;
    row_.Reshape(row_shape);
    abs_row_.Reshape(row_shape);
    row_sign_.Reshape(row_shape);
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "row_.shape: " << row_.shape_string();
    vector<int> _2d_q_shape;
    _2d_q_shape.push_back(height - 1);
    _2d_q_shape.push_back(height);
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "_2d_q_shape: " << _2d_q_shape[0]
    //    << ", " << _2d_q_shape[1];
    _2d_q_.Reshape(_2d_q_shape);
    Dtype* _2d_q_data = _2d_q_.mutable_cpu_data();
    caffe_set(_2d_q_.count(), (Dtype)0., _2d_q_data);

    for (int i = 0; i < _2d_q_shape[0]; ++i) {
        caffe_set(1, (Dtype)1., _2d_q_data + i * _2d_q_shape[1] + i);
        caffe_set(1, (Dtype)-1., _2d_q_data + i * _2d_q_shape[1] + i + 1);
        //for (int j = 0; j < _2d_q_shape[1]; ++j) {
        //    if (i == j) {
        //        caffe_set(1, (Dtype)1., _2d_q_data + i * _2d_q_shape[0] + j);
        //    } else if (i + 1 == j) {
        //        caffe_set(1, (Dtype)-1., _2d_q_data + i * _2d_q_shape[0] + j);
        //    }
        //}
    }

    ////print_data(1, "_2d_q_", _2d_q_.cpu_data(), height - 1, height);
    
    //const Dtype* q_data = _2d_q_.cpu_data();

    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "_2d_q_"; 
    //for (int i = 0; i < height - 1; ++i) {
    //    ostringstream stream;
    //    std::string row = "";
    //    for (int j = 0; j < height; ++j) {
    //        stream << *(q_data + i * height + j) << ", ";
    //    }
    //    //LOG_IF(INFO, Caffe::root_solver())
    //        << stream.str();
    //} 
  }

  if (width > 1) {
    col_gradient_.Reshape(bottom_shape);
    //col_total_gradient.Reshape(bottom_shape);

    vector<int> col_comp_shape;
    col_comp_shape.push_back(width - 1);
    col_comp_shape.push_back(width);
    col_gradient_completer.Reshape(col_comp_shape);
    Dtype* col_gradient_data = col_gradient_completer.mutable_cpu_data();
    caffe_set(col_gradient_completer.count(), (Dtype)0., col_gradient_data);
    for (int i = 0; i < col_comp_shape[0]; ++i) {
        caffe_set(1, (Dtype)1., col_gradient_data + i * col_comp_shape[1] + i + 1);
    }
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "bottom.shape: " << bottom[0]->shape_string();
    vector<int> p_shape = bottom_shape;
    p_shape[p_shape.size() -2] = p_shape[p_shape.size() - 1];
    p_shape[p_shape.size() - 1] = p_shape[p_shape.size() - 1] -1;
    
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "p_shape: " << p_shape[0] << ", " 
    //    << p_shape[1] << ", " << p_shape[2] << ", "
    //    << p_shape[3];
    vector<int> col_shape = bottom_shape;
    col_shape[col_shape.size() - 1] -= 1;
    col_.Reshape(col_shape);
    abs_col_.Reshape(col_shape);
    col_sign_.Reshape(col_shape);
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "col_.shape: " << col_.shape_string();
    vector<int> _2d_p_shape;
    _2d_p_shape.push_back(width);
    _2d_p_shape.push_back(width - 1);
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "_2d_p_shape: " << p_shape[0]
    //    << ", " << p_shape[1];
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "p_shape: " << p_shape[0] << p_shape[1];
    _2d_p_.Reshape(_2d_p_shape);
    Dtype*  _2d_p_data = _2d_p_.mutable_cpu_data();
    caffe_set(_2d_p_.count(), (Dtype)0., _2d_p_data);
    for (int i = 0; i < _2d_p_shape[1]; ++i) {
        caffe_set(1, (Dtype)-1., _2d_p_data + i * _2d_p_shape[1] + i);
        caffe_set(1, (Dtype)1., _2d_p_data + (i + 1) * _2d_p_shape[1] + i);
        //for (int j = 0; j < _2d_p_shape[1]; ++j) {
        //    if (i == j) {
        //      caffe_set(1, (Dtype)-1., _2d_p_data + i * _2d_p_shape[0] + j); 
        //    } else if (j + 1 == i) {
        //      caffe_set(1, (Dtype)1., _2d_p_data + i * _2d_p_shape[0] + j);      
        //    }
        //}
    }
    ////print_data(1, "_2d_p_", _2d_p_.cpu_data(), width, width - 1);
    
    //const Dtype* p_data = _2d_p_.cpu_data();
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "_2d_p_"; 
    //for (int i = 0; i < width; ++i) {
    //    ostringstream stream;
    //    std::string row = "";
    //    for (int j = 0; j < width - 1; ++j) {
    //        stream << *(p_data + i * (width - 1)+ j) << ", ";
    //    }
    //    //LOG_IF(INFO, Caffe::root_solver())
    //        << stream.str();
    //} 
  }
}

//template <typename Dtype>
//void EuclideanLossWithTVLayer<Dtype>::print_data(const int num, const std::string name, const Dtype* data, int height, int width) {
//    ////LOG_IF(INFO, Caffe::root_solver())
//    //    << name;
//    for (int n = 0; n < num; ++n) {
//        for (int i = 0; i < height; ++i) {
//            ostringstream stream;
//            for (int j = 0; j < width; ++j) {
//              stream << *(data + n * height * width + i * width + j) << ", ";
//            }
//            ////LOG_IF(INFO, Caffe::root_solver())
//            //    << stream.str();
//        }
//        ////LOG_IF(INFO, Caffe::root_solver())
//        //    << "num: " << num;
//    }
//}



  //p_.Reshape(p_shape);
  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "p_.shape: " << p_.shape_string();
  //q_.Reshape(q_shape);
  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "q_.shape: " << q_.shape_string();
  //int num_axes = bottom[0]->num_axes();
  
  
  


template <typename Dtype>
void EuclideanLossWithTVLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  

  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "bottom[0]";
  //print_data(num_, "bottom[0]", bottom[0]->cpu_data(), height, width);
  //print_data(num_, "bottom[1]", bottom[1]->cpu_data(), height, width);
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //print_data(num_, "diff_", diff_.cpu_data(), height, width);
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  //if (height <=1 && width <= 1) {
  //  top[0]->mutable_cpu_data()[0] = loss;
  //  return;
  //}
  Dtype row_tv = 0;
  Dtype col_tv = 0;
  
  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "original loss: " << loss;
  ////print_data(num_, "bottom[0]", bottom[0]->cpu_data(), height, width);
  ////print_data(num_, "bottom[1]", bottom[1]->cpu_data(), height, width);
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (height > 1) {
    Dtype* row_data = row_.mutable_cpu_data();
    const Dtype* q_data = _2d_q_.cpu_data();
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "num_ : " << num_;
    for (int i = 0; i < num_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height - 1, width, height, 
              (Dtype)1., q_data, bottom_data + height * width * i, 
              (Dtype)0., row_data + i * (height - 1) * width);
    }
    //for (int i = 0; i < row_.count(); ++i) {
    //    Dtype square = (Dtype)0.5 *  *(row_data + i) * *(row_data + i);
    //    if (square < 1) {
    //        caffe_set(1, square, row_data + i);
    //    }
    //}
    ////LOG_IF(INFO, Caffe::root_solver())
    //    << "q * row finished";

    ////print_data(num_, "row_", row_.cpu_data(), height - 1, width);

    Dtype *abs_row_data = abs_row_.mutable_cpu_data();
    caffe_abs(num_ * (height - 1) * width , row_data, abs_row_data); 
    row_tv = caffe_cpu_asum(num_ * (height - 1) * width, abs_row_data);
    ////print_data(num_, "abs_row_", abs_row_.cpu_data(), height - 1, width);
  }

  if (width > 1) {
    Dtype* col_data = col_.mutable_cpu_data();
    const Dtype* p_data = _2d_p_.cpu_data();
    for (int i = 0; i < num_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width - 1, width, 
              (Dtype)1., bottom_data + height * width * i, p_data, (Dtype)0., 
              col_data + height * (width - 1) * i); 
    }
    //for (int i = 0; i < col_.count(); ++i) {
    //    Dtype square = (Dtype)0.5 * *(col_data + i) * *(col_data + i);
    //    if (square < 1) {
    //        caffe_set(1, square, col_data + i);
    //    }
    //}
    ////print_data(num_, "col_", col_.cpu_data(), height, width - 1);
    Dtype *abs_col_data = abs_col_.mutable_cpu_data();
    caffe_abs(num_ * height * (width - 1), col_data, abs_col_data); 
    col_tv = caffe_cpu_asum(num_ * height * (width - 1), abs_col_data);
    ////print_data(num_, "abs_col_", abs_col_.cpu_data(), height, width - 1);
  }

  Dtype tv_loss = (row_tv + col_tv) / bottom[0]->num();

  loss = loss + tv_loss;
  top[0]->mutable_cpu_data()[0] = loss;

  ////LOG_IF(INFO, Caffe::root_solver())
  //    << "top[0]->diff()[0]: " << top[0]->cpu_diff()[0];

  //LOG_IF(INFO, Caffe::root_solver())
      << "row_tv: " << row_tv << "col_tv: " << col_tv;
  //loss = loss + col_tv + row_tv;
  //LOG_IF(INFO, Caffe::root_solver())
      << "smoothed loss: " << loss;
}

template <typename Dtype>
void EuclideanLossWithTVLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  Dtype* total_gradient_data = total_gradient.mutable_cpu_data();
  Dtype* row_total_gradient_data = row_total_gradient.mutable_cpu_data();
  Dtype* col_total_gradient_data = col_total_gradient.mutable_cpu_data();
  caffe_set(total_gradient.count(), (Dtype)0., total_gradient_data);
  caffe_set(row_total_gradient.count(), (Dtype)0., row_total_gradient_data);
  caffe_set(col_total_gradient.count(), (Dtype)0., col_total_gradient_data);
  const Dtype* row_gradient_adder_data = row_gradient_adder.cpu_data();
  const Dtype* col_gradient_adder_data = col_gradient_adder.cpu_data();
  if (height > 1) {
    const Dtype* row_data = row_.cpu_data();
    Dtype* row_sign_data = row_sign_.mutable_cpu_data();
    const Dtype* row_gradient_comp_data = row_gradient_completer.cpu_data();
    Dtype* row_gradient_data = row_gradient_.mutable_cpu_data();
    for (int i = 0; i < row_.count(); ++i) {
        if (*(row_data + i) > 0) {
          caffe_set(1, (Dtype)1., row_sign_data + i);
        } else {
          caffe_set(1, (Dtype)-1., row_sign_data + i);
        }
        //if (*(row_data + i) > 1) {
        //  caffe_set(1, (Dtype)1., row_sign_data + i);
        //} else if (*(row_data + i) < -1) {
        //  caffe_set(1, (Dtype)-1., row_sign_data + i);
        //} else {
        //    const Dtype square = (Dtype)2. * *(row_data + i);
        //    caffe_sqr(1, &square, row_sign_data + i);
        //    //caffe_set(1, (Dtype)(*(row_data + i)), row_sign_data + i);
        //}
    }
    //print_data(num_, "row_", row_.cpu_data(), height - 1, width);
    //print_data(num_, "row_sign", row_sign_.cpu_data(), height - 1, width);
    //print_data(1, "row_gradient_adder", row_gradient_adder.cpu_data(), height, height);
    //print_data(1, "row_comp_data", row_gradient_comp_data, height, height - 1);
    
    //for (int i = 0; i < num_; ++i) {
    //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
    //          (Dtype)1., row_data + height * (width - 1) * i, row_gradient_comp_data, (Dtype)0., 
    //          row_gradient_data + height * width * i); 
    //  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
    //          (Dtype)1., row_gradient_data + height * width * i, row_gradient_adder_data, (Dtype)0., 
    //          row_total_gradient_data + height * width * i); 
    //}
    for (int i = 0; i < num_; ++i) {
      //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
      //        (Dtype)1., row_data + height * (width - 1) * i, row_gradient_comp_data, (Dtype)0., 
      //        row_gradient_data + height * width * i); 
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height - 1, 
              (Dtype)1., row_gradient_comp_data, row_sign_data + (height - 1) * width * i, (Dtype)0., 
              row_gradient_data + height * width * i); 
      //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
      //        (Dtype)1., row_gradient_data + height * width * i, row_gradient_adder_data, (Dtype)0., 
      //        row_total_gradient_data + height * width * i); 
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height, 
              (Dtype)1., row_gradient_adder_data, row_gradient_data + height * width * i, (Dtype)0., 
              row_total_gradient_data + height * width * i); 
    }
    //print_data(num_, "row_gradient", row_gradient_.cpu_data(), height, width);
    //print_data(num_, "row_total_gradient", row_total_gradient.cpu_data(), height, width);
  }
  if (width > 1) {
    const Dtype* col_data = col_.cpu_data();
    Dtype* col_sign_data = col_sign_.mutable_cpu_data();
    const Dtype* col_gradient_comp_data = col_gradient_completer.cpu_data();
    Dtype* col_gradient_data = col_gradient_.mutable_cpu_data();
    for (int i = 0; i < col_.count(); ++i) {
        if (*(col_data + i) > 0) {
          caffe_set(1, (Dtype)1., col_sign_data + i);
        } else {
          caffe_set(1, (Dtype)-1., col_sign_data + i);
        }
        //if (*(col_data + i) > 1) {
        //  caffe_set(1, (Dtype)1., col_sign_data + i);
        //} else if (*(col_data + i) < -1) {
        //  caffe_set(1, (Dtype)-1., col_sign_data + i);
        //} else {
        //    const Dtype square = (Dtype)2. * *(col_data + i);
        //    caffe_sqr(1, &square, col_sign_data + i);
        //    //caffe_set(1, (Dtype)(*(col_data + i)), col_sign_data + i);
        //}
    }
    //print_data(num_, "col_", col_.cpu_data(), height, width - 1);
    //print_data(num_, "col_sign", col_sign_.cpu_data(), height, width - 1);
    //print_data(1, "col_gradient_adder", col_gradient_adder.cpu_data(), width, width);
    //print_data(1, "col_comp_data", col_gradient_comp_data, width - 1, width);
    for (int i = 0; i < num_; ++i) {
      //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, height - 1, 
      //        (Dtype)1., col_gradient_comp_data, col_data + (height -1) * width * i, (Dtype)0., 
      //        col_gradient_data + height * width * i); 
      //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
      //        (Dtype)1., col_gradient_adder_data, col_gradient_data + height * width * i, (Dtype)0., 
      //        col_total_gradient_data + height * width * i); 
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width - 1, 
              (Dtype)1., col_sign_data + height * (width - 1) * i, col_gradient_comp_data, (Dtype)0., 
              col_gradient_data + height * width * i); 
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, height, width, width, 
              (Dtype)1., col_gradient_data + height * width * i, col_gradient_adder_data, (Dtype)0., 
              col_total_gradient_data + height * width * i); 
    }
    //print_data(num_, "col_gradient", col_gradient_.cpu_data(), height, width);
    //print_data(num_, "col_total_gradient", col_total_gradient.cpu_data(), height, width);
  }

  caffe_add(total_gradient.count(), row_total_gradient_data, 
          col_total_gradient_data, total_gradient_data);
  //print_data(num_, "total_gradient_data", total_gradient_data,  height, width);
  


  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      //LOG_IF(INFO, Caffe::root_solver())
          << "top.shape: " << top[0]->shape_string();
      //LOG_IF(INFO, Caffe::root_solver())
          << "bottom.shape: " << bottom[0]->shape_string();
      //LOG_IF(INFO, Caffe::root_solver())
          << "bottom.num: " << bottom[i]->num();
      //LOG_IF(INFO, Caffe::root_solver())
          << "bottom.count: " << bottom[i]->count();
      //LOG_IF(INFO, Caffe::root_solver())
          << "i: " << i << ", top[0]->cpu_diff[0]: " << top[0]->cpu_diff()[0];
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      //LOG_IF(INFO, Caffe::root_solver())
          << "alpha: " << alpha;
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
      //print_data(num_, "bottom-diff", bottom[i]->cpu_diff(), height, width);
      //const Dtype a = top[0]->cpu_diff()[0] / bottom[i]->num();
      const Dtype a = top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_scal(total_gradient.count(), a, total_gradient_data);
      //if (sign > 0) {
      //    const Dtype a = alpha; 
      //    caffe_scal(total_gradient.count(), a, total_gradient_data);
      //} else {
      //    const Dtype a = -1 * alpha;
      //    caffe_scal(total_gradient.count(), a, total_gradient_data);
      //}
      //const Dtype a= sign / bottom[i]->num();
      //print_data(num_, "total_gradient_data", total_gradient.cpu_data(), height, width);
      //caffe_cpu_axpby(
      //    total_gradient.count(),              // count
      //    alpha,                              // alpha
      //    total_gradient.cpu_data(),                   // a
      //    Dtype(0),                           // beta
      //    total_gradient_data);  // b
      //caffe_cpu_axpby(
      //    total_gradient.count(),              // count
      //    alpha,                              // alpha
      //    total_gradient.cpu_data(),                   // a
      //    Dtype(1),                           // beta
      //    bottom[i]->mutable_cpu_diff());  // b
      ////print_data(1, "total_gradient_data", total_gradient.cpu_data(), height, width);
      ////print_data(1, "total_gradient_data", total_gradient.cpu_data(), height, width);
      caffe_axpy(total_gradient.count(), (Dtype)1, 
              total_gradient_data, bottom[i]->mutable_cpu_diff());
      //caffe_add(total_gradient.count(), total_gradient_data, 
      //        bottom[i]->cpu_diff(), bottom[i]->mutable_cpu_diff());
      //print_data(num_, i + "bottom-diff", bottom[i]->cpu_diff(), height, width);
      //LOG_IF(INFO, Caffe::root_solver())
          << "add TV gradient";
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossWithTVLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossWithTVLayer);
REGISTER_LAYER_CLASS(EuclideanLossWithTV);

}  // namespace caffe

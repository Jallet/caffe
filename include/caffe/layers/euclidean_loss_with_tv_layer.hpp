#ifndef CAFFE_EUCLIDEAN_LOSS_WITH_TV_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_WITH_TV_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Euclidean (L2) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ and TV-norm for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for smoothed least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanLossWithTVLayer exactly formulates a smoothed linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class EuclideanLossWithTVLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossWithTVLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanLossWithTV"; }
  /**
   * Unlike most loss layers, in the EuclideanLossWithTVLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossWithTVLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void print_data(const int num_, const std::string name, const Dtype *data, 
  //        int height, int width);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, EuclideanLossWithTVLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> p_;
  Blob<Dtype> q_;
  Blob<Dtype> _2d_q_; //column variation transformation matrix
  Blob<Dtype> _2d_p_; //row variation transformation matrix
  
  Blob<Dtype> row_; //row variation 
  Blob<Dtype> abs_row_; // absolute row variation
  Blob<Dtype> col_; //col variation
  Blob<Dtype> abs_col_;//absolute col variation
  
  Blob<Dtype> row_sign_;// sign of row variation
  Blob<Dtype> col_sign_;// sign of col variation
  
  Blob<Dtype> row_gradient_;// gradient of row varaition row_
  Blob<Dtype> col_gradient_; //gradient of col varaition col_ 
  
  //complete (height - 1) * width gradient to height * width gradient matrix
  Blob<Dtype> row_gradient_completer;
  //complete height * (width - 1) gradient to height * width gradient matrix
  Blob<Dtype> col_gradient_completer;
  
  Blob<Dtype> row_total_gradient;//total gradient of row, echo element is the 
                                 // sum of its left and right column
  Blob<Dtype> col_total_gradient;//total gradient of col, each element is the 
                                 //sum of its above and below row 
  Blob<Dtype> row_gradient_adder;
  Blob<Dtype> col_gradient_adder;
  //Blob<Dtype> gradient_adder;
  Blob<Dtype> total_gradient;
  //Blob<Dtype> col_gradient_adder;
  
  float lambda_;
  int num_;
  int height;
  int width;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_WITH_TV_LAYER_HPP_

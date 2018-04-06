#ifndef  _FRCNN_API_H
#define  _FRCNN_API_H
#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace FRCNN_API {
	class Detector {
	public:
		Detector(std::string &proto_file, std::string &model_file) {
			Set_Model(proto_file, model_file);
		}
		void Set_Model(std::string &proto_file, std::string &model_file);
		void predict(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results);
		void predict_original(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results);
		void predict_iterative(const cv::Mat &img_in, std::vector<caffe::Frcnn::BBox<float> > &results);
	private:
		void preprocess(const cv::Mat &img_in, const int blob_idx);
		void preprocess(const std::vector<float> &data, const int blob_idx);
		std::vector<boost::shared_ptr<caffe::Blob<float> > > predict(const std::vector<std::string> blob_names);
		boost::shared_ptr<caffe::Net<float> > net_;
		float mean_[3];
		int roi_pool_layer;
	};

}
#endif
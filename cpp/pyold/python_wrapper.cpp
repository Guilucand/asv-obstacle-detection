#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <boost/python.hpp>
#include <boost/scoped_array.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#ifdef CUDA

#ifdef JETSON
#include <opencv2/gpu/gpu.hpp>
#else
#include <opencv2/core/cuda.hpp>
#endif

#endif

using namespace cv;
using namespace boost::python;

#include "mosse.h"
#include "image_stab.h"
#include "undistort.h"
// #include "util.h"


#if (PY_VERSION_HEX >= 0x03000000)
static void *init_ar() {
#else
static void init_ar(){
#endif
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

Mat draw_horizon(Mat const& frame, Mat const& outframe, float sigma);

BOOST_PYTHON_MODULE(mosse)
{
    //using namespace XM;
    init_ar();

    class_<TransformMat>("TransformMat");

    //initialize converters
    to_python_converter<cv::Mat,
           pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    // def("divSpec2", divSpec2);
    def("draw_horizon", draw_horizon);
    def("undistort", ::undistort);
    def("canny", ::Canny);
    // def("DFT", DFT);
    // def("PDFT", PDFT);
    // def("GDFT", GDFT);
    // def("WDFT", WDFT);
    // def("GCVDFT", GCVDFT);
    class_<Point>("Point", init<int, int>())
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y);

    class_<Mosse>("Mosse", init<int, int, int, int>())
      .def("update", &Mosse::update)
      .def("reset", &Mosse::reset);

    class_<ImgStab>("ImgStab")
      .def("update", &ImgStab::update)
      .def("mat_update", &ImgStab::mat_update)
      .def("reset", &ImgStab::reset)
      .def("ftransform", &ImgStab::ftransform);
      // .def("itransform", &ImgStab::itransform);


}

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "cpp/mosse.h"
#include "cpp/image_stab.h"
#include "cpp/undistort.h"
#include "cpp/util.h"
#include "cpp/args.hxx"
#include "cpp/stdanalysis.h"

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

using namespace std;
using namespace cv;


int myErrorHandler(int status, const char* func_name, const char* err_msg,
const char* file_name, int line, void*)
{
    return 0;
}

static void affineTransform(Mat const& source, Mat &dest, Matx33f transform) {
    transform = transform.t();
    Point start = Point(0, 0);

    auto tstart = (Matx13f(start.x, start.y, 1) * transform).get_minor<1, 2>(0, 0);
    auto tnextx = (Matx13f(start.x+1, start.y, 1) * transform).get_minor<1, 2>(0, 0) - tstart;
    auto tnexty = (Matx13f(start.x, start.y+1, 1) * transform).get_minor<1, 2>(0, 0) - tstart;

    Matx12f psize = Matx12f(source.cols-1, source.rows-1);

    // cout << "Matrix: " << transform << endl;
    // cout << "Tstart: " << tstart << endl;

    Size matSize = Size(source.cols, source.rows);

    for (int i = 0; i < matSize.height; i++) {
        float *row = dest.ptr<float>(i);
        auto startr = tstart + tnexty * i;
        for (int j = 0; j < matSize.width; j++) {
            // printf("Patch (%d, %d) => (%f, %f)\n", j, i, startr(0, 0), startr(0, 1));
            row[j] = source.at<float>(clamp(startr(0, 1), 0.0f, psize(0, 1)), clamp(startr(0, 0), 0.0f, psize(0, 0)));
            startr += tnextx;
        }
    }
}


bool use_gpu;


static bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0)
{
    if(mSrc.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));

    mSrc.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst,mSrc.type());

    return true;
}

// static Mosse *tmp;
// static void CallBackFunc(int event, int x, int y, int flags, void* userdata)
// {
//
//   if ( flags == (EVENT_FLAG_LBUTTON) )
//   {
//     tmp->addWindow(Point(x, y));
//     // auto totSize = tmp->getWindowsCount();
//     // printf("Add window (%d %d)\n", x, y);
//     // for (int i = 0; i < totSize.width; i++) {
//     //   for (int j = 0; j < totSize.height; j++) {
//     //     tmp->addWindow(Point((i+1) * 16, (j+1) * 16));
//     //   }
//     // }
//   }
// }

extern Mat totframe;
extern TransformMat gstabmat;


int main(int argc, const char** argv) {
    cvRedirectError(myErrorHandler);

    args::ArgumentParser parser("Main obstacle recognition boat program\n"
#ifdef CUDA
    "Compiled with cuda support"
#endif
    );
    parser.LongSeparator(" ");

    args::Positional<string> input_(parser, "<video source>", "The input video source", args::Options::Required);

    args::HelpFlag help_(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> start_time_(parser, "start time", "Set to start elaboration from a specific time", {'t', "time"}, 0.0f);
    args::ValueFlag<float> scale_(parser, "scale", "Scale of the video, with 1 = (384x216)", {'s', "scale"}, 1.0f);
    args::ValueFlag<int> skipframes_(parser, "ratio", "Ratio (total fps)/(displayed fps)", {'k', "skip"}, 1);
    args::ValueFlag<int> horizon_stability_(parser, "hstab", "Number of training displayed frames before an horizon is trained", {'l', "learn-time"}, 25);
    args::ValueFlag<string> output_(parser, "output", "Set to save elaboration to file", {'o', "output"}, "");

    args::ValueFlag<string> exp_name_(parser, "exp name", "Experiment name", {'e', "exp"}, "");

    args::Flag fps_sync_(parser, "fps sync", "Set to fix video speed", {'f', "fix-fps"}, false);
    args::Flag pause_(parser, "pause", "Set to start video as paused", {'p', "pause"}, false);

#ifdef PROFILER
    args::ValueFlag<string> profile_(parser, "<output profile>", "Start program in profile mode", {'d', "debug"}, "");
#endif

#ifdef CUDA
    args::Flag gpu_(parser, "gpu", "Set to use cuda for elaboration", {'g', "gpu"}, false);
#endif

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Completion e)
    {
        cout << e.what();
        return 0;
    }
    catch (args::Help)
    {
        cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        cerr << e.what() << endl;
        cerr << parser;
        return 1;
    }
    catch (args::RequiredError e)
    {
        cerr << e.what() << endl;
        cerr << parser;
        return 1;
    }

    string file_name = string(args::get(input_));
    bool pause = args::get(pause_);
    float start_time = args::get(start_time_);
    float scale = args::get(scale_);
    string outfile_name = args::get(output_);
    bool fps_sync = args::get(fps_sync_);

#ifdef PROFILER
    string prof_output = args::get(profile_);
    bool profile = prof_output != "";
#endif

#ifdef CUDA
    use_gpu = args::get(gpu_);
#else
    use_gpu = false;
#endif

    string exp_name = args::get(exp_name_);

    int skip_frames = args::get(skipframes_);

    VideoCapture cap = VideoCapture(file_name);
    cap.set(CV_CAP_PROP_POS_MSEC, start_time * 1000);
    int length = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
    float fps = cap.get(CV_CAP_PROP_FPS);

    int totwidth = 384 * scale;
    float ratio = cap.get(CV_CAP_PROP_FRAME_HEIGHT) / cap.get(CV_CAP_PROP_FRAME_WIDTH);
    Point2i defsize = Point2i(totwidth, totwidth * ratio);

    bool undistortImg = false;

    VideoWriter out_video;
    bool out_open = false;
    if (outfile_name != "") {
        int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
        out_video = VideoWriter(outfile_name, ex, cap.get(CV_CAP_PROP_FPS) / skip_frames, Size(defsize.x, defsize.y));
        if (!out_video.isOpened()) {
            printf("Failed to open file '%s' for writing", outfile_name.c_str());
        }
        else
        out_open = true;
    }

    int val = args::get(horizon_stability_);//40;//
    int horizon_stability = val*2;
    int horizon_validity = val;

    Mosse horizon_detector = Mosse(32, 32, defsize.x, defsize.y);
    // Mosse horizon_detector_2 = Mosse(32, 32, defsize.x, defsize.y);
    ImgStab cpp_imstab = ImgStab();
    TransformMat stabmat = TransformMat();
    // TransformMat stabmat_2 = TransformMat();
    horizon_detector.since_reset = horizon_validity;
    // horizon_detector_2.since_reset = 0;

    // tmp = &horizon_detector;

    int frame_idx = -1;

    auto last_time = chrono::high_resolution_clock::now();
    bool onestep = false;
    int reset = 0;
    bool black = false;
    Mat last_frame, frame, frame_raw;


    bool single_frame_tracking = false;

    // oldoverlay = None

    bool first = true;
    // previmtouse = None

    //Create a window
    namedWindow("stab_frame", 1);

    //set the callback function for any mouse event
    // setMouseCallback("stab_frame", CallBackFunc, NULL);

    // Mat resized_frame = np.zeros(defsize, np.uint8)

    bool reset1, reset2;

#ifdef PROFILER
    if (profile)
    ProfilerStart(prof_output.c_str());
#endif

    Mat to_disp;
    Mat canny_uint;
    // Mat framestab;
    // Mat framestab1_full, framestab2_full;
    // Mat framestab1_color, framestab2_color;
    Mat can1, can2;
    Mat canny, graysc;
    Mat anmat;

    Mat horizMask;

    Rect roi;
    int current_frame = 0;
    int percent = -1;
    int real_fps = 0, fps_counter = 0;
    auto fps_timer = chrono::high_resolution_clock::now();
    while (cap.isOpened()) {

        bool resetan = false;
        reset1 = reset2 = false;
        frame_idx += 1;

        int new_percent = (current_frame+1) * 100 / length;
        if (new_percent != percent && new_percent >= 0 && new_percent <= 100) {
            percent = new_percent;
            printf("Complete at %d%%\n", percent);
        }
        // printf("*********************** NEW FRAME ***********************\n");

        auto elapsed_time = chrono::high_resolution_clock::now() - last_time;
        int ms = chrono::duration_cast<chrono::milliseconds>(elapsed_time).count();
        int us = chrono::duration_cast<chrono::microseconds>(elapsed_time).count();
        if (fps_sync) {
            int to_sleep = 1000000/fps - us/skip_frames;
            if (to_sleep > 0) {
                auto stime = chrono::microseconds(to_sleep);
                this_thread::sleep_for(stime);
                // sleep(max(0, 1.0/fps - elapsed_time))
                elapsed_time += stime;
            }
        }
        last_time = chrono::high_resolution_clock::now();

        auto no_overhead_start_time = chrono::high_resolution_clock::now();

        if (!pause || first) {

            for (int i = 0; i < skip_frames-1; i++) {
                current_frame++;
                cap.read(frame_raw);
            }

            bool flag = cap.read(frame_raw);


            current_frame++;
            no_overhead_start_time = chrono::high_resolution_clock::now();

            if (!flag)
            break; // Exit w/o error/s

            resize(frame_raw, frame, Size { defsize.x, defsize.y });
            if (undistortImg)
            undistort(frame);

            // GaussianBlur(frame, frame, Size(0,0), 0.8) ;
            // AddGaussianNoise_Opencv(frame, frame, 0.0f, 2.4f);

            cvtColor(frame, graysc, COLOR_BGR2GRAY);

            graysc.copyTo(totframe);
            graysc.convertTo(canny, CV_32F);
            canny.copyTo(anmat);
            // GaussianBlur(canny, canny, Size(0,0), 0.8);

            Canny(canny);
            imshow("Canny", canny/256.0);

            // imwrite("origframe.png", frame);
            // imwrite("canny.png", canny);


            // Mat tmp2 = canny/256.0f;
            // tmp2.convertTo(canny_uint, CV_8U);
            // log(tmp2*8192.0+1.0, tmp2);
            // double min, max;
            // minMaxLoc(tmp2, &min, &max, NULL, NULL);
            // imshow("Canny", ((tmp2 - min)/(max-min)));
            // minMaxLoc(tmp2, &min, &max, NULL, NULL);
            // printf("MinMax: %lf %lf\n", min, max);
            // cvtColor(canny_uint, canny_uint, COLOR_GRAY2BGR);

            float psr_th = 12.0;

            // if (horizon_detector.since_reset >= horizon_stability+1) {
            //     reset1 = true;
            //     cpp_imstab.reset(stabmat);
            //     cout << "Reset1!" << endl;
            // }
            // if (horizon_detector_2.since_reset >= horizon_stability+1) {
            //     reset2 = true;
            //     cpp_imstab.reset(stabmat_2);
            //     cout << "Reset2!" << endl;
            // }

            // horizon_detector.displaydbg(Point(200, 200));
            // Mat startframe = frame;

            // int totsize = sqrt(pow(defsize.x,2)+pow(defsize.y,2)) + 32;
            // int pad[2] = { 0, 0 };
            // int hpad1[2] = { pad[0]/2, pad[1]/2 };
            // int hpad2[2] = { pad[0]/2, pad[1]/2 };
            //
            //
            // copyMakeBorder(canny, framestab1_full, 0, pad[1], 0, pad[0], BORDER_REPLICATE);
            // framestab1_full.copyTo(framestab2_full);


            // copyMakeBorder(frame, framestab1_color, 0, pad[1], 0, pad[0], BORDER_CONSTANT);
            // framestab1_color.copyTo(framestab2_color);

            cpp_imstab.update(graysc+0.0f);//DEBUG

            TransformMat stabmat_std;

            cpp_imstab.mat_update(stabmat);
            cpp_imstab.mat_update(stabmat_std);


            Mat test;
            frame.copyTo(test);
            TransformMat tmp1;
            cpp_imstab.ftransform(tmp1, test, false, 2, false);
            // imwrite("lk-tracking.png", test);
            // imshow("Test", test);

            // cpp_imstab.ftransform(stabmat, framestab1_color, true, 0, true, Point(hpad1[0], hpad1[1]));
            // cpp_imstab.ftransform(stabmat_2, framestab2_color, true, 0, true, Point(hpad2[0], hpad2[1]));

            // Point disp1, disp2;
            // disp1 = cpp_imstab.ftransform(stabmat, framestab1_full, false, 0, false, Point(hpad1[0], hpad1[1]));
            // disp2 = cpp_imstab.ftransform(stabmat_2, framestab2_full, false, 0, false, Point(hpad2[0], hpad2[1]));

            horizMask = Mat::zeros(frame.rows, frame.cols, CV_32FC4);


            // Mat test;
            // canny.copyTo(test);
            // imshow("Test", test/256.0);
            //
            // Mat test1 = Mat::zeros(canny.rows, canny.cols, CV_32FC1);
            // affineTransform(canny, test1, stabmat.trinv);
            // imshow("Test1", test1/256.0);

            Mat res1 = analyze(anmat+0.0f, reset==2, cpp_imstab);

            static deque<Mat> last_mats;

            static Point displ;

            Mat res1c = res1 + 0.0f;
            displ = cpp_imstab.ftransform(stabmat_std, res1c, false, 0, false, Point(0, 0));
            // last_mats.push_back(res1);
            // last_mats.push_back(res1);
            // last_mats.push_back(res1);


            while (last_mats.size() > 1) {
                last_mats.pop_front();
            }
            //
            //

            for (auto mat : last_mats) {
                res1c += mat;
            }
            inRange(res1c, 0.5, 1.5, res1c);

            // cpp_imstab.ftransform(gstabmat, horizMask, true, 0, false);
            // blend(res1c, horizMask);
            int erosion_size = 2;
            Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

           Mat tmpc = res1c+0.0f;
           imshow("AnalysisB", tmpc);

#if 0
            /// Apply the erosion operation
            erode( res1c, res1c, element );
            dilate( res1c, res1c, element );
            dilate( res1c, res1c, element );
            dilate( res1c, res1c, element );

            imshow("AnalysisA", res1c);
            last_mats.clear();
            last_mats.push_back(res1+0.0f);

            imshow("AnalysisC", tmpc-res1c);


            Mat tmp1;
            frame.convertTo(tmp1, CV_32FC3);
            Mat tmp2;
            Mat tmp4;

            vector<Mat> channels;
            channels.push_back(res1);
            channels.push_back(res1);
            channels.push_back(res1);

            merge(channels, tmp2);
            merge(channels, tmp4);

            tmp4 = Scalar(1.0f, 1.0f, 1.0f) - tmp4;

            Mat tmp3;
            multiply(tmp4, tmp1, tmp3, 1.0f/256.0f);


            multiply(tmp2, tmp1, tmp1, 1.0f/256.0f);
            imshow("TEST1", tmp1);


            imshow("TEST2", tmp3);

            // tmp1 *= 256.0f;
            // tmp3 *= 256.0f;

            // imwrite("filter1.png", tmp1);
            // imwrite("filter2.png", tmp3);
#endif
            static int counter = 0;
            if (!(counter++ % 10)) {
                horizon_detector.refresh(res1, stabmat);
            }


            if (reset == 2) {
                resetan = true;
                horizon_detector.reset(res1, stabmat);
                horizon_detector.reset_history();
                reset = 0;
            }

            horizon_detector.update(canny, horizMask, psr_th, stabmat);


            horizon_detector.since_reset +=1;

            frame.copyTo(to_disp);
            if (black)
            to_disp *= 0;
            blend(to_disp, horizMask);

            auto no_overhead_elapsed_time = chrono::high_resolution_clock::now() - no_overhead_start_time;
            int no_overhead_ms = chrono::duration_cast<chrono::milliseconds>(no_overhead_elapsed_time).count();

            // draw_text(to_disp, to_string(real_fps * skip_frames), Point(20, 50));//1000/max(1,ms) * skip_frames

            if (out_open) {
                out_video << to_disp;
            }
        }

        bool takepic = false;
        fps_counter++;
        auto fps_elapsed_time = chrono::high_resolution_clock::now() - fps_timer;
        int total_ms_fps = chrono::duration_cast<chrono::milliseconds>(fps_elapsed_time).count();
        // if (total_ms_fps >= 1000) {
        //     fps_timer = chrono::high_resolution_clock::now();
        //     real_fps = fps_counter;
        //     fps_counter = 0;
        //     takepic = true;
        // }

        imshow("stab_frame", to_disp);

        last_frame = frame;
        int ch = waitKey(1);
        if (onestep) {
            onestep = false;
            pause = true;
        }
        switch (ch) {
        case'r':
            reset = 2;
            if (pause) {
                pause = false;
                onestep = true;
            }
            break;
        case'u':
            undistortImg = !undistortImg;
            break;
        case 'c':
            reset = 1;
            single_frame_tracking = false;
            if (pause) {
                pause = false;
                onestep = true;
            }
            break;
        case 's':
            reset = 1;
            single_frame_tracking = true;
            if (pause) {
                pause = false;
                onestep = true;
            }
            break;
        case 'b':
            black = !black;
            pause = false;
            onestep = true;
            break;
        case 32:
            pause = !pause;
            break;
        case 'f':
            pause = false;
            onestep = true;
            break;

        case 'a': {
            takepic = true;
        }
        break;
    }
    if (ch == 27)
    break;

    if (takepic && exp_name.size()>0) {
        static int widx = 0;
        imwrite(string("../output/images/")+exp_name+to_string(widx++)+".png", to_disp);
        printf("Taken picture %d => %s\n", widx-1, exp_name.c_str());
    }

    first = false;

    static int mod = 0;
    if (false){//(!pause && (mod++ % 40 == 0) && exp_name.size()>0) {
        for (auto &win : horizon_detector.mosseWindows) {
            static int goodidx = 0;
            static int badidx = 0;
            int maximgs = 50000;
            float prob = 0.1f;

            if (rand() >= RAND_MAX*prob) continue;

            if (goodidx >= maximgs && badidx >= maximgs) exit(0);
            if (win->alive_time > 15) {
                cv::Point pos = win->position;
                auto tpos = (cv::Matx13f(pos.x, pos.y, 1) * stabmat.trinv.t()).get_minor<1, 2>(0, 0);
                pos = Point(tpos(0, 0), tpos(0, 1));
                if (pos.y < 20 || pos.x < 20 || pos.y >= frame.rows-20 || pos.x >= frame.cols-20)
                continue;

                Mat tmp = Mat::zeros(32, 32, CV_8UC3);
                extractSubMatTransform<Vec3b>(frame, Size(32, 32), win->position, tmp, stabmat.trinv.t());
                if (win->mean_goodtime > 15 && goodidx < maximgs) {
                    string str = string("../output/trainimg/obstc/")+exp_name+to_string(goodidx++)+".png";
                    imwrite(str, tmp);
                }
                else if (badidx < maximgs && win->perm_deleted && pos.y > frame.rows/2){
                    string str = string("../output/trainimg/water/")+exp_name+to_string(badidx++)+".png";
                    imwrite(str, tmp);
                }
            }
        }
    }
}
#ifdef PROFILER
if (profile)
    ProfilerStop();
#endif
    destroyAllWindows();
}

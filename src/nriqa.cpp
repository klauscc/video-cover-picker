#include "nriqa.h"

int nriqa::findSaArea(Mat im) {
    if (cv::sum(im).val[0] > 0) {
        im = im / cv::sum(im).val[0];
    }
    vector<float> hv;
    for (int i = 0; i < im.cols; i++) {
        float val = 0;
        for (int j = 0; j < im.rows; j++) {
            val += im.at<float>(j,i);
        }
        hv.push_back(val);
    }
    vector<float> score_v;
    for (int i = 0; i < im.cols - im.cols / 2 + 1; i++) {
        float val = 0;
        for (int j = 0; j < im.cols / 2; j++) {
            val += hv[i+j];
        }
        score_v.push_back(val);
    }
    float max_v = 0;
    int max_index = 0;
    for (int i = 0; i < score_v.size(); i++) {
        if (max_v < score_v[i]) {
            max_v = score_v[i];
            max_index = i;
        }
    }
    return max_index;
}

void nriqa::nriqaInit(){
    min_img_height = 256;
    min_img_width = 256;
    area_w_size = 9;
    area_h_size = 9;
    perception_mask = (cv::Mat_<float>(area_w_size, area_h_size) << 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0,
                                                                    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                                                    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
                                                                    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
                                                                    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
                                                                    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
                                                                    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8,
                                                                    0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                                                                    0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0);
}

vector<float> nriqa::nriqaQualityIndex(Mat img, Mat vsi) {
    /*---------- small size image ----------*/
    int height, width;
    width = img.cols;
    height = img.rows;
    if (height < min_img_height || width < min_img_width) {
        std::cout << " Image is too small to satisfy model's request [" << min_img_height << "*" << min_img_width << "]" << std::endl;
        exit(1);
    }
    //namedWindow("input img", CV_WINDOW_AUTOSIZE);
    //imshow("input img", img);
    /*---------- convert to gray image ----------*/
    Mat I, Ir, G, Gr, P, E;
    float g_mssim, g_entropy, nrss, Rp, Rh;
    if (img.channels() == 4)
        cvtColor(img, I, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 3)
        cvtColor(img, I, cv::COLOR_BGR2GRAY);
    else
        I = img;

    I.convertTo(I, CV_32FC1);                                        // I
    GaussianBlur(I, Ir, cv::Size(7, 7), cv::sqrt(6), cv::sqrt(6));   // Ir
    G = sobelFilter(I);                                              // G
    Gr = sobelFilter(Ir);                                            // Gr
    P = G.mul(G);                                                    // P
    E = cannyEdgeCalc(I);                                            // E


    g_mssim = calcMSSIM(G, Gr);
    g_entropy = calcImgEntropy(I);
    Mat Rp_mat =  calcRatioPower(E, P);
    Mat nrss_mat = calcImgNrss(G, Gr);
    Mat Rh_mat = calcImgHitRatio(vsi);
    Rp_mat = Rp_mat.mul(Rh_mat);
    nrss_mat = nrss_mat.mul(Rh_mat);

    int sal_col = findSaArea(Rh_mat);
    Mat Rh_mat_transpose;
    transpose(Rh_mat, Rh_mat_transpose);
    int sal_row = findSaArea(Rh_mat_transpose);

    Rp = cv::sum(Rp_mat(Rect(sal_col, sal_row, area_w_size / 2, area_h_size / 2))).val[0];
    nrss = cv::sum(nrss_mat(Rect(sal_col, sal_row, area_w_size / 2, area_h_size / 2))).val[0];

    /*{
        if (cv::sum(Rh_mat).val[0] > 0) {
            Rh_mat = Rh_mat / cv::sum(Rh_mat).val[0];
            vector<float> hv;
            for (int i = 0; i < Rh_mat.cols; i++) {
                float val = 0;
                for (int j = 0; j < Rh_mat.rows; j++) {
                    val += Rh_mat.at<float>(j,i);
                }
                hv.push_back(val);
            }
            float s0, s1, s2, s3, s4, s5;
            s0 = hv[0] + hv[1] + hv[2] + hv[3];
            s1 = hv[1] + hv[2] + hv[3] + hv[4];
            s2 = hv[2] + hv[3] + hv[4] + hv[5];
            s3 = hv[3] + hv[4] + hv[5] + hv[6];
            s4 = hv[4] + hv[5] + hv[6] + hv[7];
            s5 = hv[5] + hv[6] + hv[7] + hv[8];


            float hv_l = hv[0] + hv[1] + hv[2];
            float hv_c = hv[2] + hv[3] + hv[4] + hv[5] + hv[6];
            float hv_r = hv[6] + hv[7] + hv[8];


            if (hv_c <= hv_l && hv_c <= hv_r) {
                Rh = (hv_l + hv_r) * 0.8;
            }

            if ((hv_l - hv_c) * (hv_r - hv_c) <= 0) {
                Rh = 0.9*(hv_l > hv_r ? hv_l : hv_r);
            }

            if (hv_c >= hv_l && hv_c >= hv_r) {
                Rh = hv_c;
            }


        }else {
            Rh = 0;
        }
    }*/

    std::cout << "global mssim is: " << g_mssim << std::endl;
    std::cout << "nrss is: " << nrss << std::endl;
    std::cout << "Rp is: " << Rp << std::endl;
    //std::cout << "Rh is: " << Rh << std::endl;
    //std::cout << "global entropy is: " << g_entropy << std::endl;
    //transpose(Rp_mat, Rp_mat);
    //transpose(nrss_mat, nrss_mat);
    //transpose(Rh_mat, Rh_mat);
    //std::cout << "Rp_mat is: \n " << Rp_mat << std::endl;
    //std::cout << "nrss_mat is: \n " << nrss_mat << std::endl;
    //std::cout << "Rh_mat is: \n " << Rh_mat << std::endl;
    //std::cout << cv::sum(Rp_mat).val[0] << std::endl;

    //G.convertTo(G, CV_8UC1);
    //namedWindow("G img", CV_WINDOW_AUTOSIZE);
    //imshow("G img", G);
    //P.convertTo(P, CV_8UC1);
    //namedWindow("Power img", CV_WINDOW_AUTOSIZE);
    //imshow("Power img", P);
    //namedWindow("edge img", CV_WINDOW_AUTOSIZE);
    //imshow("edge img", E);

    float iqa;
    float weight = 0.55;
    float weight2 = 0.7;
    iqa = (1 - weight) * (weight2 * nrss + (1 - weight2) * g_mssim) + weight * Rp;
    vector<float> qaIndex;
    qaIndex.push_back(iqa);
    qaIndex.push_back(nrss);
    qaIndex.push_back(g_mssim);
    qaIndex.push_back(Rp);
    //qaIndex.push_back(Rh);
    //qaIndex.push_back(g_entropy);
    return qaIndex;
}

Mat nriqa::sobelFilter(Mat im) {
    Mat sobelX = (cv::Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
    Mat sobelY = (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    sobelX = sobelX / 4;
    sobelY = sobelY / 4;
    Mat edgeX, edgeY;
    filter2D(im, edgeX, im.type(), sobelX);
    filter2D(im, edgeY, im.type(), sobelY);
    Mat gradMag;
    cv::sqrt((edgeX.mul(edgeX) + edgeY.mul(edgeY)), gradMag);
    return gradMag;
}

Mat nriqa::cannyEdgeCalc(Mat im) {
    im.convertTo(im, CV_8UC1);
    int edge_threshould = 50;
    Mat im_edge;
    Canny(im, im_edge, edge_threshould, edge_threshould*3, 3);

    //namedWindow("edge img", CV_WINDOW_AUTOSIZE);
    //imshow("edge img", im_edge);

    im_edge = im_edge / 255;
    im_edge.convertTo(im_edge, CV_32FC1);

    //double maxIntensity, minIntensity;
    //cv::minMaxLoc(im_edge, &minIntensity, &maxIntensity, NULL, NULL);
    //std::cout << "max value is: " << maxIntensity << ", min value is: " << minIntensity << std::endl;
    return im_edge;
}

float nriqa::calcMSSIM(Mat I1, Mat I2) {
    const double c1 = 6.5025, c2 = 58.5225;
    Mat I1_I1, I2_I2, I1_I2, mu1, mu2, mu1_mu1, mu2_mu2, mu1_mu2, sigma1_2, sigma2_2, sigma12;

    I1_I1 = I1.mul(I1);
    I2_I2 = I2.mul(I2);
    I1_I2 = I1.mul(I2);

    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    mu1_mu1 = mu1.mul(mu1);
    mu2_mu2 = mu2.mul(mu2);
    mu1_mu2 = mu1.mul(mu2);

    GaussianBlur(I1_I1, sigma1_2, cv::Size(11, 11), 1.5);
    GaussianBlur(I2_I2, sigma2_2, cv::Size(11, 11), 1.5);
    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_mu1;
    sigma2_2 -= mu2_mu2;
    sigma12 -= mu1_mu2;

    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + c1;
    t2 = 2 * sigma12 + c2;
    t3 = t1.mul(t2);
    t1 = mu1_mu1 + mu2_mu2 + c1;
    t2 = sigma1_2 +sigma2_2 + c2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    Scalar mssim = cv::mean(ssim_map);

    return 1.0 - (1.0 <= mssim.val[0] ? 1.0 : mssim.val[0]);
}

float nriqa::calcImgEntropy(Mat im) {
    im.convertTo(im, CV_8UC1);

    const int channels[1] = {0};
    const int histSize[1] = {32};
    float pranges[] = {0, 255};
    const float* ranges[1] = {pranges};
    MatND hist;

    cv::calcHist(&im, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
    hist.convertTo(hist, CV_32FC1);
    int h = im.rows;
    int w = im.cols;
    float entropy = 0;
    for (int i = 0; i < hist.cols; i++) {
        if (hist.at<float>(0,i) != 0) {
            float prob = hist.at<float>(0,i) / h / w;
            entropy += -prob * cv::log(prob);
        }
    }

    return entropy;
}

Mat nriqa::calcImgNrss(Mat im1, Mat im2) {
    int h, w;
    h = im1.rows;
    w = im1.cols;
    int area_h = h / area_h_size;
    int area_w = w / area_w_size;

    Mat nrss_mat(area_w_size, area_h_size, CV_32FC1);
    for (int i = 0; i < area_h_size; i++) {
        for (int j = 0; j < area_w_size; j++) {
            //std::cout << (i-1)*area_w << ", " << (j-1)*area_h << std::endl;
            Mat im1_part = im1(Rect(j*area_w, i*area_h, area_w, area_h));
            Mat im2_part = im2(Rect(j*area_w, i*area_h, area_w, area_h));

            nrss_mat.at<float>(i, j) = calcMSSIM(im1_part, im2_part);
        }
    }
    return nrss_mat;
}

Mat nriqa::calcImgHitRatio(Mat im) {
    int h, w;
    h = im.rows;
    w = im.cols;
    int area_h = h / area_h_size;
    int area_w = w / area_w_size;
    Mat Rh_mat(area_w_size, area_h_size, CV_32FC1);
    float total_val = cv::sum(im).val[0];
    for (int i = 0; i < area_h_size; i++) {
        for (int j = 0; j < area_w_size; j++) {
            Mat im_part = im(Rect(j*area_w, i*area_h, area_w, area_h));
            float p = cv::sum(im_part).val[0];
            if (total_val <= 0) {
                Rh_mat.at<float>(i, j) = 0;
            }else {
                Rh_mat.at<float>(i, j) = p / total_val;
            }

        }
    }
    Rh_mat = Rh_mat.mul(perception_mask);
    return Rh_mat;
}

Mat nriqa::calcRatioPower(Mat edge, Mat power) {
    int h, w;
    h = edge.rows;
    w = edge.cols;
    int area_h = h / area_h_size;
    int area_w = w / area_w_size;
    Mat Rp_mat(area_w_size, area_h_size, CV_32FC1);

    for (int i = 0; i < area_h_size; i++) {
        for (int j = 0; j < area_w_size; j++) {
            Mat edge_part = edge(Rect(j*area_w, i*area_h, area_w, area_h));
            Mat power_part = power(Rect(j*area_w, i*area_h, area_w, area_h));

            float p = cv::sum(power_part).val[0];
            if (p <= 0) {
                Rp_mat.at<float>(i, j) = 0;
            }else {
                float pe = cv::sum(power_part.mul(edge_part)).val[0];
                Rp_mat.at<float>(i, j) = pe / p;
            }

        }
    }
    return Rp_mat;
}

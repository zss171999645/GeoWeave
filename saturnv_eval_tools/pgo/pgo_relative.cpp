// optimize_pose_graph_singlethread.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <set>

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace fs = std::filesystem;
using namespace std;

// ----------------------------- Data structures -----------------------------
struct Pose {
    double timestamp{};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

struct RelativeConstraint {
    string cam_i;
    int fid_i;
    string cam_j;
    int fid_j;
    Eigen::Vector3d t_ij;
    Eigen::Quaterniond R_ij;
    double weight;
};

// ----------------------------- Cost functors -----------------------------
struct UnifiedRelativePoseError {
public:
    UnifiedRelativePoseError(const Eigen::Vector3d& t_ij_meas,
                             const Eigen::Quaterniond& R_ij_meas,
                             double weight)
        : t_ij_(t_ij_meas), R_ij_(R_ij_meas), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const trans_f_i,
                    const T* const quat_f_i,
                    const T* const trans_f_j,
                    const T* const quat_f_j,
                    const T* const extrin_t_i,   // t_fc: front->cam_i
                    const T* const extrin_q_i,   // q_fc_i
                    const T* const extrin_t_j,   // t_fc: front->cam_j
                    const T* const extrin_q_j,   // q_fc_j
                    const T* const scale,
                    T* residuals) const
    {
        // front poses
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_f_i(trans_f_i);
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_f_j(trans_f_j);
        Eigen::Quaternion<T> q_f_i(quat_f_i[3], quat_f_i[0], quat_f_i[1], quat_f_i[2]);
        Eigen::Quaternion<T> q_f_j(quat_f_j[3], quat_f_j[0], quat_f_j[1], quat_f_j[2]);
        q_f_i.normalize(); q_f_j.normalize();

        // extrinsic front->cam_i
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_fc_i(extrin_t_i);
        Eigen::Quaternion<T> q_fc_i(extrin_q_i[3], extrin_q_i[0], extrin_q_i[1], extrin_q_i[2]);
        q_fc_i.normalize();

        // extrinsic front->cam_j
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_fc_j(extrin_t_j);
        Eigen::Quaternion<T> q_fc_j(extrin_q_j[3], extrin_q_j[0], extrin_q_j[1], extrin_q_j[2]);
        q_fc_j.normalize();

        // Compose T_wc_i = T_wf_i * T_fc_i
        Eigen::Quaternion<T> q_wc_i = q_f_i * q_fc_i;
        Eigen::Matrix<T,3,1> t_wc_i = t_f_i + q_f_i * t_fc_i;

        // Compose T_wc_j = T_wf_j * T_fc_j
        Eigen::Quaternion<T> q_wc_j = q_f_j * q_fc_j;
        Eigen::Matrix<T,3,1> t_wc_j = t_f_j + q_f_j * t_fc_j;

        // Relative in cam-i frame
        Eigen::Quaternion<T> q_ij_est = q_wc_i.conjugate() * q_wc_j;
        Eigen::Matrix<T,3,1> t_ij_est = q_wc_i.conjugate() * (t_wc_j - t_wc_i);

        // Compare to measured
        Eigen::Quaternion<T> q_ij_meas_inv = R_ij_.cast<T>().conjugate();
        Eigen::Quaternion<T> q_err = q_ij_meas_inv * q_ij_est;
        q_err.normalize();
        // 小角度近似：对于小旋转，2 * vec(q) ≈ 旋转向量
        Eigen::Matrix<T,3,1> rot_res = T(2.0) * q_err.vec();

        Eigen::Matrix<T,3,1> t_res = t_ij_est - scale[0] * t_ij_.cast<T>();

        residuals[0] = T(weight_*100) * rot_res[0];
        residuals[1] = T(weight_*100) * rot_res[1];
        residuals[2] = T(weight_*100) * rot_res[2];
        residuals[3] = T(weight_) * t_res[0];
        residuals[4] = T(weight_) * t_res[1];
        residuals[5] = T(weight_) * t_res[2];
        return true;
    }

private:
    const Eigen::Vector3d t_ij_;
    const Eigen::Quaterniond R_ij_;
    const double weight_;
};

// 处理相同相机情况的代价函数
struct SameCameraRelativePoseError {
public:
    SameCameraRelativePoseError(const Eigen::Vector3d& t_ij_meas,
                                const Eigen::Quaterniond& R_ij_meas,
                                double weight)
        : t_ij_(t_ij_meas), R_ij_(R_ij_meas), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const trans_f_i,
                    const T* const quat_f_i,
                    const T* const trans_f_j,
                    const T* const quat_f_j,
                    const T* const extrin_t,   // t_fc: front->cam
                    const T* const extrin_q,   // q_fc
                    const T* const scale,
                    T* residuals) const
    {
        // front poses
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_f_i(trans_f_i);
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_f_j(trans_f_j);
        Eigen::Quaternion<T> q_f_i(quat_f_i[3], quat_f_i[0], quat_f_i[1], quat_f_i[2]);
        Eigen::Quaternion<T> q_f_j(quat_f_j[3], quat_f_j[0], quat_f_j[1], quat_f_j[2]);
        q_f_i.normalize(); q_f_j.normalize();

        // extrinsic front->cam
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_fc(extrin_t);
        Eigen::Quaternion<T> q_fc(extrin_q[3], extrin_q[0], extrin_q[1], extrin_q[2]);
        q_fc.normalize();

        // Compose T_wc_i = T_wf_i * T_fc
        Eigen::Quaternion<T> q_wc_i = q_f_i * q_fc;
        Eigen::Matrix<T,3,1> t_wc_i = t_f_i + q_f_i * t_fc;

        // Compose T_wc_j = T_wf_j * T_fc
        Eigen::Quaternion<T> q_wc_j = q_f_j * q_fc;
        Eigen::Matrix<T,3,1> t_wc_j = t_f_j + q_f_j * t_fc;

        // Relative in cam-i frame
        Eigen::Quaternion<T> q_ij_est = q_wc_i.conjugate() * q_wc_j;
        Eigen::Matrix<T,3,1> t_ij_est = q_wc_i.conjugate() * (t_wc_j - t_wc_i);

        // Compare to measured
        Eigen::Quaternion<T> q_ij_meas_inv = R_ij_.cast<T>().conjugate();
        Eigen::Quaternion<T> q_err = q_ij_meas_inv * q_ij_est;
        q_err.normalize();
        // 小角度近似：对于小旋转，2 * vec(q) ≈ 旋转向量
        Eigen::Matrix<T,3,1> rot_res = T(2.0) * q_err.vec();

        Eigen::Matrix<T,3,1> t_res = t_ij_est - scale[0] * t_ij_.cast<T>();

        residuals[0] = T(weight_*100) * rot_res[0];
        residuals[1] = T(weight_*100) * rot_res[1];
        residuals[2] = T(weight_*100) * rot_res[2];
        residuals[3] = T(weight_) * t_res[0];
        residuals[4] = T(weight_) * t_res[1];
        residuals[5] = T(weight_) * t_res[2];
        return true;
    }

private:
    const Eigen::Vector3d t_ij_;
    const Eigen::Quaterniond R_ij_;
    const double weight_;
};

// ----------------------------- Helpers -----------------------------
static bool ReadTumPoses(const fs::path& tum_file, std::vector<Pose>& out, double init_scale) {
    if (!fs::exists(tum_file)) return false;
    ifstream fin(tum_file);
    string line;
    out.clear();
    out.reserve(64);

    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        Pose p;
        ss >> p.timestamp
           >> p.t.x() >> p.t.y() >> p.t.z()
           >> p.q.x() >> p.q.y() >> p.q.z() >> p.q.w();
        if (ss.fail()) continue;
        p.q.normalize();
        p.t *= init_scale;
        out.push_back(p);
    }
    return !out.empty();
}

static void SavePoses(const string& filename,
                      const map<int, Pose>& global_poses) {
    ofstream fout(filename);
    fout << fixed << setprecision(9);
    for (const auto& kv : global_poses) {
        const Pose& p = kv.second;
        fout << p.timestamp << " "
             << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
             << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << "\n";
    }
    fout.close();
}

// 解析图像文件名，提取相机名和帧ID
static pair<string, int> ParseImageName(const string& img_name) {
    // 格式: camera_name/fid.txt
    size_t slash_pos = img_name.find('/');
    if (slash_pos == string::npos) return make_pair("", -1);
    
    string cam = img_name.substr(0, slash_pos);
    string fid_str = img_name.substr(slash_pos + 1);
    
    // 移除扩展名
    size_t dot_pos = fid_str.find('.');
    if (dot_pos != string::npos) {
        fid_str = fid_str.substr(0, dot_pos);
    }
    
    try {
        int fid = stoi(fid_str);
        return make_pair(cam, fid);
    } catch (...) {
        return make_pair("", -1);
    }
}

// 读取 pair_pose.txt 文件
static vector<RelativeConstraint> ReadPairPose(const fs::path& pair_file) {
    vector<RelativeConstraint> constraints;
    if (!fs::exists(pair_file)) return constraints;
    
    ifstream fin(pair_file);
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string img1, img2;
        double score, x, y, z, qx, qy, qz, qw;
        int segment_id;
        
        ss >> segment_id >> img1 >> img2 >> score >> x >> y >> z >> qx >> qy >> qz >> qw;
        
        auto cam_fid1 = ParseImageName(img1);
        auto cam_fid2 = ParseImageName(img2);
        
        if (cam_fid1.second < 0 || cam_fid2.second < 0) continue;
        
        RelativeConstraint c;
        c.cam_i = cam_fid1.first;
        c.fid_i = cam_fid1.second;
        c.cam_j = cam_fid2.first;
        c.fid_j = cam_fid2.second;
        c.t_ij = Eigen::Vector3d(x, y, z);
        c.R_ij = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        c.weight = score;
        
        constraints.push_back(c);
    }
    return constraints;
}

// 读取 vggt_cam2w_scaled.txt 文件的第一帧位姿
static bool ReadFirstPoseFromVggt(const fs::path& vggt_file, Pose& pose, double scale) {
    if (!fs::exists(vggt_file)) return false;
    ifstream fin(vggt_file);
    string line;
    if (!getline(fin, line)) return false;
    
    stringstream ss(line);
    ss >> pose.timestamp
       >> pose.t.x() >> pose.t.y() >> pose.t.z()
       >> pose.q.x() >> pose.q.y() >> pose.q.z() >> pose.q.w();
    if (ss.fail()) return false;
    
    pose.q.normalize();
    pose.t *= scale;
    return true;
}

// 计算外参
static void ComputeExtrinsic(const Pose& pose_front, const Pose& pose_cam, 
                             Eigen::Vector3d& t_fc, Eigen::Quaterniond& q_fc) {
    // T_fc = T_wf^{-1} * T_wc
    Eigen::Matrix3d R_wf = pose_front.q.toRotationMatrix();
    Eigen::Matrix3d R_wc = pose_cam.q.toRotationMatrix();
    Eigen::Matrix3d R_fc = R_wf.transpose() * R_wc;
    t_fc = R_wf.transpose() * (pose_cam.t - pose_front.t);
    q_fc = Eigen::Quaterniond(R_fc);
    q_fc.normalize();
}

// ----------------------------- Main -----------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "用法: " << argv[0] << " <数据目录> <相机1> <相机2> ... --front=<前置相机>" << endl;
        return 1;
    }


    const string data_dir = argv[1];
    const double kInitScale = 1.0;
    const bool   kUseHuberLoss = true;
    const double kHuberDelta = 1.0;

    // 多相机集合
    vector<string> kCams;
    string kFront;

    // 从 argv[2] 开始解析
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg.rfind("--front=", 0) == 0) {
            kFront = arg.substr(8); // 提取 --front=xxx
        } else {
            kCams.push_back(arg);
        }
    }

    if (kFront.empty()) {
        cerr << "必须指定 --front=<前置相机>" << endl;
        return 1;
    }

    // 调试输出
    cout << "数据目录: " << data_dir << "\n";
    cout << "相机集合: ";
    for (auto& c : kCams) cout << c << " ";
    cout << "\n前置相机: " << kFront << endl;

    // 存储所有约束
    vector<RelativeConstraint> all_constraints;

    // 存储所有帧ID（按相机分组）
    map<string, set<int>> cam_frame_ids;

    // 存储所有位姿（按相机和帧ID）
    map<string, map<int, Pose>> global_poses;

    // 读取所有 pair_pose.txt 文件
    fs::path pair_pose_dir = fs::path(data_dir) / "pair_pose";
    if (!fs::exists(pair_pose_dir) || !fs::is_directory(pair_pose_dir)) {
        cerr << "错误: pair_pose 目录不存在: " << pair_pose_dir << endl;
        return 1;
    }

    // 遍历所有 segment 目录
    for (const auto& seg_entry : fs::directory_iterator(pair_pose_dir)) {
        if (!seg_entry.is_directory()) continue;
        
        fs::path pair_file = seg_entry.path() / "pair_pose.txt";
        if (!fs::exists(pair_file)) continue;
        std::cout << "读取 " << pair_file << "..." << std::endl;
        auto constraints = ReadPairPose(pair_file);
        all_constraints.insert(all_constraints.end(), constraints.begin(), constraints.end());
        
        // 收集帧ID
        for (const auto& c : constraints) {
            cam_frame_ids[c.cam_i].insert(c.fid_i);
            cam_frame_ids[c.cam_j].insert(c.fid_j);
        }
    }

    if (all_constraints.empty()) {
        cerr << "错误: 没有找到任何约束" << endl;
        return 1;
    }

    cout << "共读取约束数: " << all_constraints.size() << endl;


    struct Segment {
        int start_id{};
        int end_id{};
        vector<Pose> poses;
    };

    // 每个相机的 segments（读取 vggt，用于预对齐 & 约束）
    map<string, vector<Segment>> cam_segments;
    // front 的全局 pose（用于优化、输出）
    map<int, Pose> global_front_poses;

    // 读取四个相机的 vggt segments
    for (const auto& cam : kCams) {
        fs::path cam_dir = fs::path(data_dir) / cam;
        if (!fs::exists(cam_dir) || !fs::is_directory(cam_dir)) {
            cerr << "[WARN] 相机目录不存在: " << cam_dir << "\n";
            continue;
        }
        vector<Segment> segments;
        for (const auto& entry : fs::directory_iterator(cam_dir)) {
            if (!entry.is_directory()) continue;
            string dir_name = entry.path().filename().string();
            size_t dash = dir_name.find('-');
            if (dash == string::npos) continue;

            int start_id = 0, end_id = 0;
            try {
                start_id = stoi(dir_name.substr(0, dash));
                end_id   = stoi(dir_name.substr(dash + 1));
            } catch (...) {
                cerr << "跳过无法解析的目录名: " << dir_name << " (cam=" << cam << ")\n";
                continue;
            }

            fs::path tum = entry.path() / "vggt_cam2w.txt";
            vector<Pose> poses;
            if (!ReadTumPoses(tum, poses, kInitScale)) {
                cerr << "警告: 读不到或为空: " << tum << "\n";
                continue;
            }
            if ((int)poses.size() != (end_id - start_id)) {
                cerr << "提示: " << cam << "/" << dir_name << " 行数=" << poses.size()
                     << " 期望=" << (end_id - start_id) << "（继续）\n";
            }
            segments.push_back(Segment{start_id, end_id, std::move(poses)});
        }
        if (segments.empty()) {
            cerr << "[WARN] 相机无有效数据: " << cam << "\n";
        } else {
            sort(segments.begin(), segments.end(),
                [](const Segment& a, const Segment& b){ return a.start_id < b.start_id; });
        }
        cam_segments[cam] = std::move(segments);
    }

    if (cam_segments[kFront].empty()) {
        cerr << "没有有效 front 数据，退出。\n";
        return 1;
    }

    // 打印统计
    size_t totalSegs=0;
    for (auto& kv : cam_segments) totalSegs += kv.second.size();
    cout << "共读取相机数: " << cam_segments.size() << ", 总分段: " << totalSegs << endl;

    // ---------------- 预对齐 (SE3, 按相机分别对齐) ----------------
    auto se3_align_segments = [](vector<Segment>& segments, const string& cam_tag){
        if (segments.size() < 2) return;
        for (size_t seg_idx = 1; seg_idx < segments.size(); ++seg_idx) {
            const auto& ref_seg = segments[seg_idx - 1];
            auto& cur_seg = segments[seg_idx];

            int overlap_start = max(ref_seg.start_id, cur_seg.start_id);
            int overlap_end   = min(ref_seg.end_id,   cur_seg.end_id);
            if (overlap_end <= overlap_start) continue;

            vector<Eigen::Vector3d> pts_ref, pts_cur;
            vector<Eigen::Quaterniond> qs_ref, qs_cur;
            for (int fid = overlap_start; fid < overlap_end; ++fid) {
                const Pose& p_ref = ref_seg.poses[fid - ref_seg.start_id];
                const Pose& p_cur = cur_seg.poses[fid - cur_seg.start_id];
                pts_ref.push_back(p_ref.t);
                pts_cur.push_back(p_cur.t);
                qs_ref.push_back(p_ref.q);
                qs_cur.push_back(p_cur.q);
            }
            if (pts_ref.size() < 3) continue;

            // 平均旋转
            Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < qs_ref.size(); ++i) {
                M += qs_ref[i].toRotationMatrix() * qs_cur[i].toRotationMatrix().transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R_align = svd.matrixU() * svd.matrixV().transpose();
            if (R_align.determinant() < 0) {
                Eigen::Matrix3d U = svd.matrixU();
                U.col(2) *= -1;
                R_align = U * svd.matrixV().transpose();
            }
            // 平移
            Eigen::Vector3d mean_src = Eigen::Vector3d::Zero();
            Eigen::Vector3d mean_dst = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < pts_cur.size(); ++i) {
                mean_src += R_align * pts_cur[i];
                mean_dst += pts_ref[i];
            }
            mean_src /= pts_cur.size();
            mean_dst /= pts_ref.size();
            Eigen::Vector3d t_align = mean_dst - mean_src;

            // 应用
            for (auto& p : cur_seg.poses) {
                p.t = R_align * p.t + t_align;
                p.q = Eigen::Quaterniond(R_align) * p.q;
                p.q.normalize();
            }
            cout << "[对齐] " << cam_tag << " Segment " << seg_idx
                 << " 已对齐到 Segment " << (seg_idx-1) << endl;
        }
    };

    for (auto& kv : cam_segments) {
        se3_align_segments(kv.second, kv.first);
    }

    // ---------------- front 初始位姿（对齐后） ----------------
    for (const auto& seg : cam_segments[kFront]) {
        for (int i = 0; i < (int)seg.poses.size(); ++i) {
            int fid = seg.start_id + i;
            global_poses[kFront][fid] = seg.poses[i];
        }
    }





    // // ---------------- 初始化位姿 ----------------
    // // 所有位姿初始化为原点
    // for (const auto& cam : kCams) {
    //     if (cam_frame_ids.find(cam) == cam_frame_ids.end()) continue;
        
    //     for (int fid : cam_frame_ids[cam]) {
    //         Pose p;
    //         p.timestamp = fid * 0.1; // 假设时间戳
    //         p.t = Eigen::Vector3d::Zero();
    //         p.q = Eigen::Quaterniond::Identity();
    //         global_poses[cam][fid] = p;
    //     }
    // }

    // ---------------- 初始化外参 ----------------
    struct ExtrinParam {
        array<double, 3> t;
        array<double, 4> q;
    };
    map<string, ExtrinParam> extrin_params;

    // 为 front 相机设置单位外参
    extrin_params[kFront] = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0} // 单位四元数
    };

    // 读取 front 相机第一帧位姿
    Pose front_first_pose;
    bool has_front_first_pose = false;
    for (const auto& seg_entry : fs::directory_iterator(fs::path(data_dir) / kFront)) {
        if (!seg_entry.is_directory()) continue;
        
        fs::path vggt_file = seg_entry.path() / "odo_cam2w.txt";
        if (ReadFirstPoseFromVggt(vggt_file, front_first_pose, kInitScale)) {
            has_front_first_pose = true;
            break;
        }
    }

    if (!has_front_first_pose) {
        cerr << "错误: 无法读取 front 相机第一帧位姿" << endl;
        return 1;
    }

    // 为每个非 front 相机计算外参
    for (const auto& cam : kCams) {
        if (cam == kFront) continue;
        
        // 查找该相机的第一个 segment
        Pose cam_first_pose;
        bool has_cam_first_pose = false;
        for (const auto& seg_entry : fs::directory_iterator(fs::path(data_dir) / cam)) {
            if (!seg_entry.is_directory()) continue;
            
            fs::path vggt_file = seg_entry.path() / "odo_cam2w.txt";
            if (ReadFirstPoseFromVggt(vggt_file, cam_first_pose, kInitScale)) {
                has_cam_first_pose = true;
                break;
            }
        }
        
        if (!has_cam_first_pose) {
            cerr << "警告: " << cam << " 无法读取第一帧位姿" << endl;
            continue;
        }
        
        // 计算外参 T_fc = T_wf^{-1} * T_wc
        Eigen::Vector3d t_fc;
        Eigen::Quaterniond q_fc;
        ComputeExtrinsic(front_first_pose, cam_first_pose, t_fc, q_fc);
        
        extrin_params[cam] = {
            {t_fc.x(), t_fc.y(), t_fc.z()},
            {q_fc.x(), q_fc.y(), q_fc.z(), q_fc.w()}
        };
        
        cout << "初始化外参: " << cam << " -> front" << endl;
        cout << "  t_fc: " << t_fc.transpose() << endl;
        cout << "  q_fc: " << q_fc.x() << ", " << q_fc.y() << ", " << q_fc.z() << ", " << q_fc.w() << endl;
    }

    // ---------------- Ceres Problem ----------------
    ceres::Problem problem;

    // 参数块存储
    map<string, map<int, array<double, 3>>> trans_params;
    map<string, map<int, array<double, 4>>> rot_params;
    
    // 添加参数块
    for (const auto& cam : kCams) {
        if (global_poses.find(cam) == global_poses.end()) continue;
        
        for (auto& kv : global_poses[cam]) {
            int fid = kv.first;
            Pose& p = kv.second;
            
            trans_params[cam][fid] = {p.t.x(), p.t.y(), p.t.z()};
            rot_params[cam][fid] = {p.q.x(), p.q.y(), p.q.z(), p.q.w()};
            
            auto* quat_param = new ceres::EigenQuaternionParameterization();
            problem.AddParameterBlock(rot_params[cam][fid].data(), 4, quat_param);
            problem.AddParameterBlock(trans_params[cam][fid].data(), 3);
        }
    }

    // 添加外参参数块
    for (const auto& cam : kCams) {
        if (cam == kFront) continue;
        if (extrin_params.find(cam) == extrin_params.end()) continue;
        
        auto* quat_param = new ceres::EigenQuaternionParameterization();
        problem.AddParameterBlock(extrin_params[cam].q.data(), 4, quat_param);
        problem.AddParameterBlock(extrin_params[cam].t.data(), 3);
        
        // 固定外参
        problem.SetParameterBlockConstant(extrin_params[cam].q.data());
        problem.SetParameterBlockConstant(extrin_params[cam].t.data());
    }

    // 添加尺度参数并固定为1.0
    map<string, double> cam_scales;
    for (const auto& cam : kCams) {
        cam_scales[cam] = 1.0;
        problem.AddParameterBlock(&cam_scales[cam], 1);
        problem.SetParameterBlockConstant(&cam_scales[cam]); // 固定为1.0
    }

    // 固定 front 第一帧
    if (!global_poses[kFront].empty()) {
        int first_frame = global_poses[kFront].begin()->first;
        problem.SetParameterBlockConstant(trans_params[kFront][first_frame].data());
        problem.SetParameterBlockConstant(rot_params[kFront][first_frame].data());
        cout << "固定第一帧: " << kFront << "/" << first_frame << endl;
    }

    // ---------------- 添加残差块 ----------------
    for (const auto& c : all_constraints) {
        // 检查参数块是否存在
        if (trans_params.find(kFront) == trans_params.end() ||
            trans_params[kFront].find(c.fid_i) == trans_params[kFront].end() ||
            trans_params[kFront].find(c.fid_j) == trans_params[kFront].end()) {
            cerr << "警告: 缺少参数块: " << kFront << "/" << c.fid_i 
                 << " 或 " << kFront << "/" << c.fid_j << endl;
            continue;
        }
        
        // 检查外参是否存在
        if (extrin_params.find(c.cam_i) == extrin_params.end() ||
            extrin_params.find(c.cam_j) == extrin_params.end()) {
            cerr << "警告: 缺少外参: " << c.cam_i << " 或 " << c.cam_j << endl;
            continue;
        }
        
        ceres::LossFunction* loss_function = nullptr;
        if (kUseHuberLoss) {
            loss_function = new ceres::HuberLoss(kHuberDelta);
        }
        
        // 处理相同相机的情况
        if (c.cam_i == c.cam_j) {
            // 使用单独的代价函数处理相同相机的情况
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<SameCameraRelativePoseError, 6, 3, 4, 3, 4, 3, 4, 1>(
                    new SameCameraRelativePoseError(c.t_ij, c.R_ij, c.weight));
            
            problem.AddResidualBlock(cost_function, loss_function,
                trans_params[kFront][c.fid_i].data(),
                rot_params[kFront][c.fid_i].data(),
                trans_params[kFront][c.fid_j].data(),
                rot_params[kFront][c.fid_j].data(),
                extrin_params[c.cam_i].t.data(),
                extrin_params[c.cam_i].q.data(),
                &cam_scales[c.cam_i]);
        } else {
            // 使用统一的代价函数处理不同相机的情况
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<UnifiedRelativePoseError, 6, 3, 4, 3, 4, 3, 4, 3, 4, 1>(
                    new UnifiedRelativePoseError(c.t_ij, c.R_ij, c.weight));
            
            problem.AddResidualBlock(cost_function, loss_function,
                trans_params[kFront][c.fid_i].data(),
                rot_params[kFront][c.fid_i].data(),
                trans_params[kFront][c.fid_j].data(),
                rot_params[kFront][c.fid_j].data(),
                extrin_params[c.cam_i].t.data(),
                extrin_params[c.cam_i].q.data(),
                extrin_params[c.cam_j].t.data(),
                extrin_params[c.cam_j].q.data(),
                &cam_scales[c.cam_i]);
        }
    }

    cout << "添加残差块总数: " << problem.NumResidualBlocks() << endl;
    cout << "开始求解..." << endl;

    ceres::Solver::Options options;
    options.num_threads = 1;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_type = ceres::TRUST_REGION;
    options.max_num_iterations = 500;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;

    // ---------------- 更新优化后的位姿 ----------------
    for (const auto& cam : kCams) {
        if (global_poses.find(cam) == global_poses.end()) continue;
        
        for (auto& kv : global_poses[cam]) {
            int fid = kv.first;
            const auto& t_arr = trans_params[cam][fid];
            const auto& q_arr = rot_params[cam][fid];
            
            kv.second.t = Eigen::Vector3d(t_arr[0], t_arr[1], t_arr[2]);
            kv.second.q = Eigen::Quaterniond(q_arr[3], q_arr[0], q_arr[1], q_arr[2]);
            kv.second.q.normalize();
        }
    }

    // 对于非 front 相机，使用 front 相机位姿和外参计算最终位姿
    for (const auto& cam : kCams) {
        if (cam == kFront) continue;
        if (global_poses.find(cam) == global_poses.end()) continue;
        
        // 检查外参是否存在
        if (extrin_params.find(cam) == extrin_params.end()) {
            cerr << "警告: 缺少外参: " << cam << endl;
            continue;
        }
        
        // 获取外参
        const auto& ep = extrin_params[cam];
        Eigen::Vector3d t_fc(ep.t[0], ep.t[1], ep.t[2]);
        Eigen::Quaterniond q_fc(ep.q[3], ep.q[0], ep.q[1], ep.q[2]);
        q_fc.normalize();
        
        // 更新每个帧的位姿
        for (auto& kv : global_poses[cam]) {
            int fid = kv.first;
            Pose& p = kv.second;
            
            // 获取优化后的 front 相机位姿
            if (global_poses[kFront].find(fid) == global_poses[kFront].end()) {
                cerr << "警告: front 相机缺少帧: " << fid << endl;
                continue;
            }
            
            const Pose& front_pose = global_poses[kFront][fid];
            
            // 计算该相机的位姿: T_wc = T_wf * T_fc
            p.q = front_pose.q * q_fc;
            p.t = front_pose.t + front_pose.q * t_fc;
            p.q.normalize();
        }
    }

    // ---------------- 输出结果 ----------------
    fs::path out_dir = fs::path(data_dir) / "pgo_results";
    fs::create_directories(out_dir);
    
    // 输出每个相机的轨迹
    for (const auto& cam : kCams) {
        if (global_poses.find(cam) == global_poses.end()) continue;
        
        fs::path out_file = out_dir / (cam + "_optimized.txt");
        ofstream fout(out_file);
        fout << fixed << setprecision(9);
        
        // 按帧ID排序
        vector<pair<int, Pose>> sorted_poses;
        for (const auto& kv : global_poses[cam]) {
            sorted_poses.push_back({kv.first, kv.second});
        }
        sort(sorted_poses.begin(), sorted_poses.end(), 
            [](const pair<int, Pose>& a, const pair<int, Pose>& b) {
                return a.first < b.first;
            });
        
        for (const auto& kv : sorted_poses) {
            const Pose& p = kv.second;
            fout << p.timestamp << " "
                 << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
                 << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << "\n";
        }
        fout.close();
        cout << "已写入: " << out_file << endl;
    }

    // 输出外参
    ofstream fext(out_dir / "optimized_extrinsics.txt");
    fext << fixed << setprecision(9);
    for (const auto& cam : kCams) {
        if (cam == kFront) continue;
        if (extrin_params.find(cam) == extrin_params.end()) continue;
        
        const auto& ep = extrin_params[cam];
        fext << cam << " "
             << ep.t[0] << " " << ep.t[1] << " " << ep.t[2] << " "
             << ep.q[0] << " " << ep.q[1] << " " << ep.q[2] << " " << ep.q[3] << "\n";
    }
    fext.close();
    cout << "已写入: " << out_dir / "optimized_extrinsics.txt" << endl;

    cout << "优化完成，结果保存在: " << out_dir << endl;
    return 0;
}
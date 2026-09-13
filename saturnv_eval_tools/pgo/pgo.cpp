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

// ----------------------------- Cost functors -----------------------------
// front 相机内部两帧相对约束
struct ScaleRelativePoseError {
public:
    ScaleRelativePoseError(const Eigen::Vector3d& t_ij_meas,
                           const Eigen::Quaterniond& R_ij_meas,
                           double weight)
        : t_ij_(t_ij_meas), R_ij_(R_ij_meas), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const trans_i,
                    const T* const quat_i,
                    const T* const trans_j,
                    const T* const quat_j,
                    const T* const scale,
                    T* residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_i(trans_i);
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_j(trans_j);

        Eigen::Quaternion<T> q_i(quat_i[3], quat_i[0], quat_i[1], quat_i[2]);
        Eigen::Quaternion<T> q_j(quat_j[3], quat_j[0], quat_j[1], quat_j[2]);
        q_i.normalize();
        q_j.normalize();

        // rotation residual
        Eigen::Quaternion<T> q_ij_est = q_i.conjugate() * q_j;
        Eigen::Quaternion<T> q_ij_meas_inv = R_ij_.cast<T>().conjugate();
        Eigen::Quaternion<T> q_err = q_ij_meas_inv * q_ij_est;
        q_err.normalize();
// 小角度近似：对于小旋转，2 * vec(q) ≈ 旋转向量
Eigen::Matrix<T,3,1> rot_res = T(2.0) * q_err.vec();

        // translation residual
        Eigen::Matrix<T,3,1> t_res = q_i.conjugate() * (t_j - t_i) - scale[0] * t_ij_.cast<T>();

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

// 非 front 相机：用 front 的两帧 + 外参 预测该相机相对运动
struct RelPoseErrorWithExtrinsic {
public:
    RelPoseErrorWithExtrinsic(const Eigen::Vector3d& t_ij_meas,
                              const Eigen::Quaterniond& R_ij_meas,
                              double weight)
        : t_ij_(t_ij_meas), R_ij_(R_ij_meas), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const trans_f_i,
                    const T* const quat_f_i,
                    const T* const trans_f_j,
                    const T* const quat_f_j,
                    const T* const extrin_t_fc,   // t_fc: front->cam
                    const T* const extrin_q_fc,   // q_fc
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
        Eigen::Map<const Eigen::Matrix<T,3,1>> t_fc(extrin_t_fc);
        Eigen::Quaternion<T> q_fc(extrin_q_fc[3], extrin_q_fc[0], extrin_q_fc[1], extrin_q_fc[2]);
        q_fc.normalize();

        // Compose T_wc = T_wf * T_fc
        Eigen::Quaternion<T> q_wc_i = q_f_i * q_fc;
        Eigen::Quaternion<T> q_wc_j = q_f_j * q_fc;
        Eigen::Matrix<T,3,1> t_wc_i = t_f_i + q_f_i * t_fc;
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


static std::vector<int> read_int_array(const fs::path& filename) {
    std::ifstream file(filename);
    std::vector<int> data;
    int value;
    
    while (file >> value) {
        data.push_back(value);
    }
    
    return data;
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

// NEW: 从某个相机目录下，扫描所有 segment 目录，读取给定文件名，构建 fid->Pose 映射（不修改 poses，仅为取外参）
static map<int, Pose> BuildPoseMapFromCamDir(const fs::path& cam_dir,
                                             const string& filename,
                                             double scale = 1.0) {
    map<int, Pose> m;
    if (!fs::exists(cam_dir) || !fs::is_directory(cam_dir)) return m;

    for (const auto& entry : fs::directory_iterator(cam_dir)) {
        if (!entry.is_directory()) continue;
        string dir_name = entry.path().filename().string();

        int segment_id = stoi(dir_name);
        fs::path fpath = entry.path() / filename;
        vector<Pose> poses;
        if (!ReadTumPoses(fpath, poses, scale)) continue;
        m[segment_id] = poses[0];
    }
    return m;
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
    const double kConstraintWeight = 1.0;
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
    

    struct Segment {
        int seg_id{};
        vector<int> img_ids;
        vector<Pose> poses;
        bool is_loop_closure{false};
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
            fs::path img_idxes = entry.path() / "idxes.txt";
            std::vector<int> img_ids = read_int_array(img_idxes);
            sort(img_ids.begin(), img_ids.end());
            // img_ids是否连续
            bool is_loop_closure = false;
            for (int i = 1; i < img_ids.size(); ++i) {
                if (img_ids[i] != img_ids[i-1] + 1) {
                    is_loop_closure = true;
                    break;
                }
            }

            fs::path tum = entry.path() / "vggt_cam2w.txt";
            vector<Pose> poses;
            if (!ReadTumPoses(tum, poses, kInitScale)) {
                cerr << "警告: 读不到或为空: " << tum << "\n";
                continue;
            }

            if ((int)poses.size() != img_ids.size()) {
                cerr << "提示: " << cam << "/" << dir_name << " 行数=" << poses.size()
                     << " 期望=" << img_ids.size() << "（继续）\n";
            }
            segments.push_back(Segment{stoi(dir_name), std::move(img_ids), std::move(poses), is_loop_closure});
        }
        if (segments.empty()) {
            cerr << "[WARN] 相机无有效数据: " << cam << "\n";
        } else {
            sort(segments.begin(), segments.end(),
                [](const Segment& a, const Segment& b){ return a.img_ids.front() < b.img_ids.front(); });
        }
        std::cout << "相机 " << cam << " 读取 " << segments.size() << " 个分段\n";
        std::cout << "其中 " << count_if(segments.begin(), segments.end(),
            [](const Segment& seg){ return seg.is_loop_closure; }) << " 个是 loop closure\n";
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

            vector<int> overlap;
            set_intersection(ref_seg.img_ids.begin(),ref_seg.img_ids.end(),cur_seg.img_ids.begin(),cur_seg.img_ids.end(),back_inserter(overlap));//求交集 
            // std::cout << "refseg: ";
            // for (int i: ref_seg.img_ids) {
            //     std::cout << i << " ";
            // }
            // std::cout <<"curseg: ";
            // for (int i: cur_seg.img_ids) {
            //     std::cout << i << " ";
            // }
            // std::cout << "相机" << cam_tag << "分段" << seg_idx - 1 << "与分段" << seg_idx << "重叠图像数:" << overlap.size() << std::endl;
            vector<Eigen::Vector3d> pts_ref, pts_cur;
            vector<Eigen::Quaterniond> qs_ref, qs_cur;
            for (int fid: overlap) {
                const Pose& p_ref = ref_seg.poses[std::distance(ref_seg.img_ids.begin(), find(ref_seg.img_ids.begin(), ref_seg.img_ids.end(), fid))];
                const Pose& p_cur = cur_seg.poses[std::distance(cur_seg.img_ids.begin(), find(cur_seg.img_ids.begin(), cur_seg.img_ids.end(), fid))];
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
            // cout << "[对齐] " << cam_tag << " Segment " << seg_idx
            //      << " 已对齐到 Segment " << (seg_idx-1) << endl;
        }
    };

    for (auto& kv : cam_segments) {
        se3_align_segments(kv.second, kv.first);
    }

    // ---------------- front 初始位姿（对齐后） ----------------
    for (const auto& seg : cam_segments[kFront]) {
        for (int i = 0; i < (int)seg.poses.size(); ++i) {
            int fid = seg.img_ids[i];
            global_front_poses[fid] = seg.poses[i];
        }
    }
    // SavePoses("initial_poses_front.txt", global_front_poses);
    // cout << "[DEBUG] 已写 initial_poses_front.txt (对齐后)" << endl;

    // ---------------- 为初始化外参，读取每个相机的 odo 轨迹并构建 fid->pose 映射 ----------------
    // NEW: 使用 odo_cam2w.txt 来计算 front->cam 外参
    map<string, map<int, Pose>> cam_sfm_pose_map;
    for (const auto& cam : kCams) {
        fs::path cam_dir = fs::path(data_dir) / cam;
        cam_sfm_pose_map[cam] = BuildPoseMapFromCamDir(cam_dir, "odo_cam2w.txt", /*scale=*/1.0);
    }

    // 也保留 vggt 的 fid->pose 映射，作为缺省回退
    auto build_pose_map = [](const vector<Segment>& segs){
        map<int, Pose> m;
        for (const auto& seg : segs) {
            for (int i = 0; i < (int)seg.poses.size(); ++i) {
                int fid = seg.img_ids[i];
                m[fid] = seg.poses[i];
            }
        }
        return m;
    };
    const map<int, Pose> front_pose_map_vggt = build_pose_map(cam_segments[kFront]);

    // ---------------- Ceres Problem ----------------
    ceres::Problem problem;

    // front 参数块
    map<int, array<double,3>> front_trans_params;
    map<int, array<double,4>> front_rot_params;
    for (auto& kv : global_front_poses) {
        int fid = kv.first;
        Pose& p = kv.second;
        front_trans_params[fid] = {p.t.x(), p.t.y(), p.t.z()};
        front_rot_params[fid]   = {p.q.x(), p.q.y(), p.q.z(), p.q.w()};
        auto* quat_local = new ceres::EigenQuaternionParameterization();
        problem.AddParameterBlock(front_rot_params[fid].data(), 4, quat_local);
        problem.AddParameterBlock(front_trans_params[fid].data(), 3);
    }

    // 每相机每 segment 的 scale（常量 1.0）
    map<string, vector<double>> cam_segment_scales;
    for (const auto& cam : kCams) {
        const auto& segs = cam_segments[cam];
        cam_segment_scales[cam] = vector<double>(segs.size(), 1.0);
        for (size_t k = 0; k < segs.size(); ++k) {
            problem.AddParameterBlock(&cam_segment_scales[cam][k], 1);
            problem.SetParameterLowerBound(&cam_segment_scales[cam][k], 0, 0.0);
            problem.SetParameterBlockConstant(&cam_segment_scales[cam][k]); // 固定
        }
    }

    // 固定 front 第一帧
    int first_frame = global_front_poses.begin()->first;
    problem.SetParameterBlockConstant(front_trans_params[first_frame].data());
    problem.SetParameterBlockConstant(front_rot_params[first_frame].data());

    // ---------------- 外参参数块（从 sfm 的第一个共同 fid 初始化，且固定） ----------------
    struct ExtrinParam { array<double,3> t; array<double,4> q; };
    map<string, ExtrinParam> extrin_params;


    ifstream fin(fs::path(data_dir) / "extrinsics.txt");
    if (!fin.is_open()) {
        cerr << "Failed to open file" << endl;
        return -1;
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;

        stringstream ss(line);

        string cam;
        ExtrinParam ep;

        ss >> cam
           >> ep.t[0] >> ep.t[1] >> ep.t[2]
           >> ep.q[0] >> ep.q[1] >> ep.q[2] >> ep.q[3];

        extrin_params[cam] = ep;
    }

    fin.close();

    for (const auto& cam : kCams) {
        if (cam == kFront) continue;
        Eigen::Vector3d t_cf(extrin_params[cam].t[0], extrin_params[cam].t[1], extrin_params[cam].t[2]);
        Eigen::Quaterniond q_cf = Eigen::Quaterniond(extrin_params[cam].q[3], extrin_params[cam].q[0], extrin_params[cam].q[1], extrin_params[cam].q[2]);
        Eigen::Matrix3d R_cf = q_cf.toRotationMatrix();
        Eigen::Matrix4d T_cf = Eigen::Matrix4d::Identity();
        T_cf.block<3,3>(0,0) = R_cf;
        T_cf.block<3,1>(0,3) = Eigen::Vector3d(t_cf[0], t_cf[1], t_cf[2]);
        Eigen::Matrix4d T_fc = T_cf.inverse();
        Eigen::Matrix3d R_fc = T_fc.block<3,3>(0,0);
        Eigen::Vector3d t_fc = T_fc.block<3,1>(0,3);
        Eigen::Quaterniond q_fc(R_fc);
        q_fc.normalize();


            ExtrinParam ep;
            ep.t = {t_fc.x(), t_fc.y(), t_fc.z()};
            ep.q = {q_fc.x(), q_fc.y(), q_fc.z(), q_fc.w()};


            cout << "[Init Extrinsic] " << cam
                 << " 初始化: t_fc=[" << t_fc.transpose() << "], "
                 << "q_fc(xyzw)=[" << q_fc.x() << "," << q_fc.y() << "," << q_fc.z() << "," << q_fc.w() << "]\n";
            cout << "[Init Extrinsic] " << cam
                 << " 初始化: t_cf=[" << t_cf.transpose() << "], "
                 << "q_cf(xyzw)=[" << q_cf.x() << "," << q_cf.y() << "," << q_cf.z() << "," << q_cf.w() << "]\n";

        
        

        extrin_params[cam] = ep;

        // 添加并固定外参参数块  // FIX: 固定外参
        auto* quat_local = new ceres::EigenQuaternionParameterization();
        problem.AddParameterBlock(extrin_params[cam].q.data(), 4, quat_local);
        problem.AddParameterBlock(extrin_params[cam].t.data(), 3);
        problem.SetParameterBlockConstant(extrin_params[cam].q.data()); // 固定旋转
        problem.SetParameterBlockConstant(extrin_params[cam].t.data()); // 固定平移
    }

    // ---------------- 残差：仅同相机内部两两约束 ----------------
    // 1) front 相机
    for (size_t seg_idx = 0; seg_idx < cam_segments[kFront].size(); ++seg_idx) {
        const auto& seg = cam_segments[kFront][seg_idx];
        const auto& poses = seg.poses;
        if (poses.size() < 2) continue;
        for (int i = 0; i < (int)poses.size(); ++i) {
            int fid_i = seg.img_ids[i];
            const auto& q_i = poses[i].q;
            const auto& t_i = poses[i].t;
            for (int j = i + 1; j < (int)poses.size(); ++j) {
                int fid_j = seg.img_ids[j];
                if (seg.is_loop_closure && std::abs(fid_i - fid_j) < 10) continue;
                const auto& q_j = poses[j].q;
                const auto& t_j = poses[j].t;
                Eigen::Quaterniond R_ij = q_i.conjugate() * q_j;
                Eigen::Vector3d    t_ij = q_i.conjugate() * (t_j - t_i);
                double frame_dist = j - i;
                if (seg.is_loop_closure) frame_dist = 5;
                double weight = kConstraintWeight/(frame_dist*frame_dist);

                ceres::CostFunction* cost =
                    new ceres::AutoDiffCostFunction<ScaleRelativePoseError, 6, 3, 4, 3, 4, 1>(
                        new ScaleRelativePoseError(t_ij, R_ij, weight));

                ceres::LossFunction* loss = nullptr;
                if (kUseHuberLoss) loss = new ceres::HuberLoss(kHuberDelta);

                problem.AddResidualBlock(cost, loss,
                    front_trans_params[fid_i].data(),
                    front_rot_params[fid_i].data(),
                    front_trans_params[fid_j].data(),
                    front_rot_params[fid_j].data(),
                    &cam_segment_scales[kFront][seg_idx]);
            }
        }
    }

    // 2) 非 front 相机：front 轨迹 + 固定外参
    for (const auto& cam : kCams) {
        double weight_cam = 0.5;
        if (cam == kFront) continue;
        if (cam == "camera_rear") weight_cam = 0.5;
        const auto& segs = cam_segments[cam];
        if (segs.empty()) continue;

        for (size_t seg_idx = 0; seg_idx < segs.size(); ++seg_idx) {
            const auto& seg = segs[seg_idx];
            const auto& poses = seg.poses;
            if (poses.size() < 2) continue;

            for (int i = 0; i < (int)poses.size(); ++i) {
                int fid_i = seg.img_ids[i];
                if (!front_trans_params.count(fid_i)) continue;

                const auto& q_i = poses[i].q;
                const auto& t_i = poses[i].t;
                for (int j = i + 1; j < (int)poses.size(); ++j) {
                    int fid_j = seg.img_ids[j];
                    if (!front_trans_params.count(fid_j)) continue;
                    if (seg.is_loop_closure && std::abs(fid_i - fid_j) < 10) continue;

                    const auto& q_j = poses[j].q;
                    const auto& t_j = poses[j].t;

                    Eigen::Quaterniond R_ij = q_i.conjugate() * q_j;
                    Eigen::Vector3d    t_ij = q_i.conjugate() * (t_j - t_i);

                    double frame_dist = j - i;
                    if (seg.is_loop_closure) frame_dist = 5;

                    double weight = kConstraintWeight*weight_cam/(frame_dist*frame_dist);

                    ceres::CostFunction* cost =
                        new ceres::AutoDiffCostFunction<RelPoseErrorWithExtrinsic, 6,
                            3,4, 3,4, 3,4, 1>(
                            new RelPoseErrorWithExtrinsic(t_ij, R_ij, weight));

                    ceres::LossFunction* loss = new ceres::CauchyLoss(kHuberDelta);

                    problem.AddResidualBlock(cost, loss,
                        front_trans_params[fid_i].data(),
                        front_rot_params[fid_i].data(),
                        front_trans_params[fid_j].data(),
                        front_rot_params[fid_j].data(),
                        extrin_params[cam].t.data(),
                        extrin_params[cam].q.data(),
                        &cam_segment_scales[cam][seg_idx]);
                }
            }
        }
    }

    cout << "开始求解..." << endl;
    cout.flush();

    ceres::Solver::Options options;
    options.num_threads = 1;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_type     = ceres::TRUST_REGION;
    options.max_num_iterations = 500;
    options.minimizer_progress_to_stdout = true;
    cout << "开始求解...1" << endl;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << "Solve() 返回，开始打印 Summary" << endl;
    cout << summary.FullReport() << endl;

    cout << "[DEBUG] Solve() 完成，准备写入结果" << endl;

    // ---------------- [FIX] 在 data_dir/pgo_pose 下输出四个相机位姿（按时间戳去重） ----------------
    {
        fs::path out_dir = fs::path(data_dir) / "before_optimization";

        auto get_front_qt = [&](int fid, Eigen::Quaterniond& q_wf, Eigen::Vector3d& t_wf){
            const auto& t_arr = front_trans_params.at(fid);
            const auto& q_arr = front_rot_params.at(fid);
            t_wf = Eigen::Vector3d(t_arr[0], t_arr[1], t_arr[2]);
            q_wf = Eigen::Quaterniond(q_arr[3], q_arr[0], q_arr[1], q_arr[2]); // (w,x,y,z)
            q_wf.normalize();
        };

        // 把 double 时间戳量化到与输出一致的 9 位小数，作为去重 key（避免浮点细微差别）
        auto ts_key = [](double ts) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(9) << ts;
            return oss.str();
        };

        for (const auto& cam : kCams) {
            // 记录已写过的时间戳 key；用 map<string,char> 代替 set 以避免新增头文件
            std::map<std::string, char> seen_ts;
            std::vector<Pose> out_poses;   // 汇总后按时间排序再写

            size_t skipped_no_front = 0;   // front 缺失的 fid（无法合成）
            size_t skipped_dup_ts   = 0;   // 重复时间戳被丢弃

            if (cam == kFront) {
                // front 直接用优化后的位姿（来自 front_trans_params/front_rot_params）
                for (const auto& seg : cam_segments.at(cam)) {
                    for (int i = 0; i < (int)seg.poses.size(); ++i) {
                        int fid = seg.img_ids[i];
                        if (!front_trans_params.count(fid)) { ++skipped_no_front; continue; }

                        const double ts = seg.poses[i].timestamp;
                        const std::string key = ts_key(ts);
                        if (!seen_ts.emplace(key, 1).second) { ++skipped_dup_ts; continue; }

                        Eigen::Quaterniond q_wf; Eigen::Vector3d t_wf;
                        get_front_qt(fid, q_wf, t_wf);

                        Pose p; p.timestamp = ts; p.t = t_wf; p.q = q_wf;
                        out_poses.push_back(std::move(p));
                    }
                }
            } else {
                // 其他相机：优化后的 front 位姿 + 固定外参 front->cam 合成世界位姿
                const auto& ep = extrin_params.at(cam);
                Eigen::Vector3d t_fc(ep.t[0], ep.t[1], ep.t[2]);
                Eigen::Quaterniond q_fc(ep.q[3], ep.q[0], ep.q[1], ep.q[2]); // (w,x,y,z)
                q_fc.normalize();

                for (const auto& seg : cam_segments.at(cam)) {
                    for (int i = 0; i < (int)seg.poses.size(); ++i) {
                        int fid = seg.img_ids[i];
                        if (!front_trans_params.count(fid)) { ++skipped_no_front; continue; }

                        const double ts = seg.poses[i].timestamp;
                        const std::string key = ts_key(ts);
                        if (!seen_ts.emplace(key, 1).second) { ++skipped_dup_ts; continue; }

                        Eigen::Quaterniond q_wf; Eigen::Vector3d t_wf;
                        get_front_qt(fid, q_wf, t_wf);

                        Eigen::Quaterniond q_wc = q_wf * q_fc;
                        Eigen::Vector3d    t_wc = t_wf + q_wf * t_fc;

                        Pose p; p.timestamp = ts; p.t = t_wc; p.q = q_wc.normalized();
                        out_poses.push_back(std::move(p));
                    }
                }
            }

            // 按时间排序再输出
            std::sort(out_poses.begin(), out_poses.end(),
                      [](const Pose& a, const Pose& b){ return a.timestamp < b.timestamp; });

            fs::path fout_path = out_dir / (cam + "_pgo.txt");
            std::ofstream fout(fout_path.string());
            fout << std::fixed << std::setprecision(9);
            for (const auto& p : out_poses) {
                fout << p.timestamp << " "
                     << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
                     << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << "\n";
            }
            fout.close();

            cout << "[DEBUG] 已写 " << fout_path
                 << " ，输出条目=" << out_poses.size()
                 << "，跳过(无 front 对齐 fid)=" << skipped_no_front
                 << "，跳过(重复时间戳)=" << skipped_dup_ts << endl;
        }
    }

    // // 输出 front 轨迹
    // {
    //     ofstream fout("optimized_poses_front_equalweight.txt");
    //     fout << fixed << setprecision(9);
    //     for (const auto& kv : global_front_poses) {
    //         int fid = kv.first;
    //         const auto& t_arr = front_trans_params[fid];
    //         const auto& q_arr = front_rot_params[fid];
    //         fout << kv.second.timestamp << " "
    //              << t_arr[0] << " " << t_arr[1] << " " << t_arr[2] << " "
    //              << q_arr[0] << " " << q_arr[1] << " " << q_arr[2] << " " << q_arr[3] << "\n";
    //     }
    //     fout.close();
    //     cout << "[DEBUG] 已写 optimized_poses_front.txt" << endl;
    // }

    // // 输出外参（front->cam），为固定值
    // {
    //     ofstream fext("optimized_extrinsics.txt");
    //     fext << fixed << setprecision(9);
    //     for (const auto& cam : kCams) {
    //         if (cam == kFront) continue;
    //         const auto& ep = extrin_params[cam];
    //         fext << cam << " "
    //              << ep.t[0] << " " << ep.t[1] << " " << ep.t[2] << " "
    //              << ep.q[0] << " " << ep.q[1] << " " << ep.q[2] << " " << ep.q[3] << "\n";
    //     }
    //     fext.close();
    //     cout << "[DEBUG] 已写 optimized_extrinsics.txt" << endl;
    // }

    // // 输出 scales（仍保持常量）
    // {
    //     ofstream fscale("optimized_scales.txt");
    //     for (const auto& cam : kCams) {
    //         for (size_t k = 0; k < cam_segment_scales[cam].size(); ++k) {
    //             fscale << cam << " " << k << " " << fixed << setprecision(9)
    //                    << cam_segment_scales[cam][k] << "\n";
    //         }
    //     }
    //     fscale.close();
    //     cout << "[DEBUG] 已写 optimized_scales.txt" << endl;
    // }

    cout << "全部完成，退出程序。" << endl;
    return 0;
}

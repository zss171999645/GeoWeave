from gtsam import CustomFactor, Values
import numpy as np

class Sim3PointCloudFactor:
    def batch_skew_symmetric(self, P):
        N = P.shape[0]
        M = np.zeros((N,3,3))
        M[:,0,1] = -P[:,2]
        M[:,0,2] =  P[:,1]
        M[:,1,0] =  P[:,2]
        M[:,1,2] = -P[:,0]
        M[:,2,0] = -P[:,1]
        M[:,2,1] =  P[:,0]
        return M

    def batch_jacobian_similarity3(self, T, P):
        s = T.scale()
        R = T.rotation().matrix()
        t = T.translation()

        N = P.shape[0]
        Rp = (R @ P.T).T
        skew_P = self.batch_skew_symmetric(P)

        J_rot = -s * np.matmul(np.broadcast_to(R, (N,3,3)), skew_P)
        J_trans = s * np.tile(R, (N,1,1))
        scaled_points = s * Rp
        J_scale = scaled_points.reshape(N,3,1)

        J = np.concatenate([J_rot, J_trans, J_scale], axis=2)
        return J

    def batch_transformFrom(self, T, points):
        s = T.scale()
        R = T.rotation().matrix()
        t = T.translation()
        return s * (points @ R.T + t)

    def error_func(self, overlap1, overlap2, this: CustomFactor, v: Values, jacobians: list[np.ndarray]=None):
        key0 = this.keys()[0]
        key1 = this.keys()[1]
        T1 = v.atSimilarity3(key0)
        T2 = v.atSimilarity3(key1)

        P1 = overlap1[:,:3]
        P2 = overlap2[:,:3]
        assert P1.shape == P2.shape, "Point clouds must have same shape"

        TP1 = self.batch_transformFrom(T1, P1)
        TP2 = self.batch_transformFrom(T2, P2)

        e = (TP1 - TP2).reshape(-1)

        if jacobians is not None:
            J1 = self.batch_jacobian_similarity3(T1, P1).reshape(-1, 7)
            J2 = self.batch_jacobian_similarity3(T2, P2).reshape(-1, 7)
            jacobians[0] = J1
            jacobians[1] = -J2
            assert J1.shape == (e.shape[0], 7)
            assert J2.shape == (e.shape[0], 7)

        return e

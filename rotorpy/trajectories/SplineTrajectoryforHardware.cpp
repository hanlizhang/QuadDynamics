kr_planning_msgs::SplineTrajectory
SplineTrajfromDiscrete( // this method will make beginning and end have an
                        // nonzero velocity issue
    const kr_planning_msgs::TrajectoryDiscretized &traj_dis_msg)
{
    int degree_plus1 = 6;
    double dt = traj_dis_msg.dt;
    kr_planning_msgs::SplineTrajectory traj_msg;
    traj_msg.header = traj_dis_msg.header;
    traj_msg.dimensions = 3;
    // prepare polynomial fit time vector
    Eigen::VectorXd time_vec = Eigen::ArrayXd::LinSpaced(degree_plus1, 0, 1);
    // create time matrix by taking this to different powers
    Eigen::MatrixXd time_mat = Eigen::MatrixXd::Zero(degree_plus1, degree_plus1);
    for (int i = 0; i < degree_plus1; i++)
    {
        time_mat.col(i) = time_vec.array().pow(i);
    }
    kr_planning_msgs::Spline spline;
    for (int dim = 0; dim < 3; dim++)
        traj_msg.data.push_back(spline);
    // ROS_INFO("Time matrix is %f", time_mat);
    for (int traj_idx = 0;
         traj_idx + (degree_plus1 - 1) < traj_dis_msg.pos.size(); // in range
         traj_idx += (degree_plus1 - 1))
    {
        // this is to have 1 repeat point to make sure things connect
        for (int dim = 0; dim < 3; dim++)
            traj_msg.data[dim].t_total = (traj_idx + (degree_plus1 - 1)) * dt;

        Eigen::MatrixXd pos_mat = Eigen::MatrixXd::Zero(degree_plus1, 3);

        // create a vector of positions
        for (int j = 0; j < degree_plus1; j++)
        {
            pos_mat(j, 0) = traj_dis_msg.pos[traj_idx + j].x;
            pos_mat(j, 1) = traj_dis_msg.pos[traj_idx + j].y;
            pos_mat(j, 2) = traj_dis_msg.pos[traj_idx + j].z;
        }
        // solve for coefficients
        Eigen::MatrixXd coeff_mat = time_mat.inverse() * pos_mat;
        // input into the new message
        for (int dim = 0; dim < 3; dim++)
        {
            traj_msg.data[dim].segments++;
            kr_planning_msgs::Polynomial p;
            p.basis = p.STANDARD;
            Eigen::VectorXd coeff_dim = coeff_mat.col(dim);
            std::vector<float> std_vector(coeff_dim.data(),
                                          coeff_dim.data() + coeff_dim.size());
            p.coeffs = std_vector;
            p.degree = degree_plus1 - 1;
            p.dt = dt * (degree_plus1 - 1);
            // p.start_index = traj_idx;
            traj_msg.data[dim].segs.push_back(p);
        }
    }

    return traj_msg;
}
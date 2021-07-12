/*
  ENG. LUCAS RAIMUNDO
  Unscented Kalman Filter - v2
*/

#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
* Initializes Unscented Kalman filter
*/

UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  // initialize covariance matrix
  P_ << 1.0, 0, 0, 0, 0,
        0, 1.0, 0, 0, 0,
        0, 0, 1000, 0, 0,
        0, 0, 0, 1000, 0,
        0, 0, 0, 0, 1.0;
        
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /*
   * End DO NOT MODIFY section for measurement noise values 
  */

  // set state dimension
  n_x_ = 5;

  // set augemented dimension
  n_aug_ = 7;

  // define spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0.0);

  // fill weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = (double)lambda_/(lambda_+n_aug_);
  for(int i=1;i<2*n_aug_+1;++i){
    weights_(i) = (double)0.5/(lambda_+n_aug_);
  }

  is_initialized_ = false;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_)
  {
    Initialize(meas_package);
  }

  double delta_t_ = (meas_package.timestamp_ - time_us_)*1e-6; // 1000000.0;
  time_us_ = meas_package.timestamp_;

  // pediction cicle
  Prediction(delta_t_);

  if (meas_package.sensor_type_ == meas_package.LASER && use_laser_)
  {
    // update lidar
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == meas_package.RADAR && use_radar_)
  {
    // update radar
    UpdateRadar(meas_package);
  }

}

void UKF::Prediction(double delta_t) {

  // augmented vector
  VectorXd x_aug = VectorXd(7); 
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // augmented covariance matrix
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  
  // augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  Xsig_aug.col(0) = x_aug;

  for(int i=0; i<n_aug_; ++i)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // sigma point prediction
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    if(fabs(yawd) > 0.01)
    {
      px_p = p_x + v/yawd * ( sin(yaw + yawd * delta_t) - sin(yaw) );
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd * delta_t) );
    }
    else
    {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p   = yaw_p   + 0.5 * nu_yawdd * delta_t*delta_t;
    yawd_p  = yawd_p  + nu_yawdd*delta_t;

    // assign predicted sigma point
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }

  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while(x_diff(3)>M_PI) x_diff(3)-=2.*M_PI;
    while(x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();

  }

  std::cout << "Prediction Result" << std::endl;
  std::cout << "x_ = \n" << x_ << std::endl;

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {

  if(is_initialized_ == true)
  {

    // measurement dimension
    int n_z = 2;

    // sigma points matrix in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);

    // mean pedicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
      // extract values 
      Zsig(0,i) = Xsig_pred_(0,i);
      Zsig(1,i) = Xsig_pred_(1,i);
    }

    // mean predicted measurement
    for(int i=0; i < 2 * n_aug_ + 1; i++)
    {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    for (int i = 0; i < 2 * n_aug_ + 1; i++) 
    {
      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;
      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_*std_laspx_, 0, 
          0, std_laspy_*std_laspy_;
    S = S + R;

    // cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) 
    {
      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain
    MatrixXd K = Tc * S.inverse();

    // raw_measurement p_x, p_y
    VectorXd z = meas_package.raw_measurements_;

    // residual
    VectorXd z_diff = z - z_pred;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    std::cout << "Update Lidar Result" << std::endl;
    std::cout << "x_ = \n" << x_ << std::endl;
 
  }

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {

  if(is_initialized_ == true)
  {

    // measurement dimension
    int n_z = 3;

    // sigma points matrix in measurement space 
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    // mean pedicted measurement
    VectorXd z_pred = VectorXd(n_z);

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);

    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
      // extract values
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double yaw = Xsig_pred_(3,i);

      double vx = cos(yaw)*v;
      double vy = sin(yaw)*v;

      double rho = sqrt(p_x*p_x+p_y*p_y);
      double phi = atan2(p_y,p_x);

      if(abs(p_x) > 0.001)
      {
        // measurement model
        phi = atan2(p_y,p_x);                           
      }
      else
      {
        // measurement model
        if(p_y > 0)
        {
          phi = M_PI/2;
        }
        else
        {
          phi = (-1)*M_PI/2;
        }
      }

      Zsig(0,i) = rho;
      Zsig(1,i) = phi;
      Zsig(2,i) = (cos(phi)*vx + sin(phi)*vy); 

    }                                    

    // mean predicted measurement
    z_pred.fill(0.0);
    for(int i=0; i < 2 * n_aug_ + 1; i++)
    {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // innovation covariance matrix S
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) 
    {
      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z,n_z);
    R <<  std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0, std_radrd_*std_radrd_;
    S = S + R;

    // cross correlation  matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) 
    {
      // residual
      VectorXd z_diff = Zsig.col(i) - z_pred;

      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      // angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // extract measured data
    VectorXd z = meas_package.raw_measurements_;

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    std::cout << "Update Radar Result" << std::endl;
    std::cout << "x_ = \n"<< x_ << std::endl;
  }
  
}

void UKF::Initialize(MeasurementPackage meas_package){

  if (meas_package.sensor_type_ == meas_package.LASER && use_laser_)
  {
    // initialization
    x_ << meas_package.raw_measurements_(0),  // p_x_0
          meas_package.raw_measurements_(1),  // p_y_0
          0,                                  // v
          0,                                  // yaw
          0;                                  // yaw
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

  }
  else if (meas_package.sensor_type_ == meas_package.RADAR && use_radar_)
  {
    // initialization
    double r    = meas_package.raw_measurements_(0); 
    double phi  = meas_package.raw_measurements_(1);
    double rd   = meas_package.raw_measurements_(2);

    while (phi> M_PI) phi-=2.*M_PI;
    while (phi<-M_PI) phi+=2.*M_PI;

    double v0 = 0.0;
    if(abs(cos(phi)) > 0.01)
    {
      v0 = rd / cos(phi);
    }

    x_ << r*cos(phi), // p_x_0
          r*sin(phi), // p_y_0
          v0,         // v_0
          0,          // yaw_0
          0;          // yawd_0
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    
  }

}

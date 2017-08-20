/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/**
 * Set the number of particles. Initialize all particles to first position (based on estimates of
 * x, y, theta and their uncertainties from GPS) and all weights to 1.
 * @param x
 * @param y
 * @param theta
 * @param std
 */
void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // Set the number of particles
    num_particles = 100;

    // Create normal distributions for x, y and theta to add random Gaussian noise to each particle.
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[3]);

    // Initialize all particles to first position (based on estimates of x, y, theta
    // and their uncertainties from GPS) and all weights to 1.
    for ( int i = 0; i < num_particles; i++ ) {

        Particle particle = { 0 };
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;

        particles.push_back(particle);
    }

    is_initialized = true;
}

/**
 * Add measurements to each particle and add random Gaussian noise.
 * @param delta_t
 * @param std_pos
 * @param velocity
 * @param yaw_rate
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    default_random_engine gen;

    for ( vector<Particle>::iterator particle = particles.begin(); particle != particles.end(); particle++ ) {

        // Add measurements to each particle

        if ( fabs(yaw_rate) < 1e-3 ) {
            // model as zero yaw rate to avoid division by zero
            double dist = velocity * delta_t;
            particle->x += dist * cos(particle->theta);
            particle->y += dist * sin(particle->theta);
        } else {
            // non-zero yaw rate
            double k = velocity / yaw_rate;
            double new_theta = particle->theta + yaw_rate * delta_t;
            particle->x += k * ( sin(new_theta) - sin(particle->theta) );
            particle->y += k * ( cos(particle->theta) - cos(new_theta) );
            particle->theta = new_theta;
        }

        // Create normal distributions for x, y and theta
        normal_distribution<double> dist_x(particle->x, std_pos[0]);
        normal_distribution<double> dist_y(particle->y, std_pos[1]);
        normal_distribution<double> dist_theta(particle->theta, std_pos[3]);

        // add random Gaussian noise
        particle->x = dist_x(gen);
        particle->y = dist_y(gen);
        particle->theta = dist_theta(gen);
    }
}

/**
 * Find the predicted measurement that is closest to each observed measurement and assign the
 * observed measurement to this particular landmark.
 * @param predicted
 * @param observations
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    for ( vector<LandmarkObs>::iterator observation = observations.begin(); observation != observations.end(); observation++ ) {

        double min_distance = numeric_limits<double>::max();
        int landmark_idx = -1;

        for ( size_t i = 0; i < predicted.size(); i++ ) {

            double distance = dist(observation->x, observation->y, predicted[i].x, predicted[i].y);
            if ( distance < min_distance ) {
                min_distance = distance;
                landmark_idx = i;
            }
        }

        if ( landmark_idx >= 0 )
            observation->id = landmark_idx;
    }
}

/**
 * Update the weights of each particle using a mult-variate Gaussian distribution.
 * @param sensor_range
 * @param std_landmark
 * @param observations
 * @param map_landmarks
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
    //
    // Pre-compute intermediate variables for Multivariate-Gaussian
    //
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double x_denominator = 2 * std_x * std_x;
    double y_denominator = 2 * std_y * std_y;
    double common_denominator = 2 * M_PI * std_x * std_y;

    //
    // Enumerate trhough the all particles
    //
    weights.clear();
    for ( vector<Particle>::iterator particle = particles.begin();
          particle != particles.end(); particle++) {

        //
        // 1. Transform observations from particle's coordinates to map's coordinates
        // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
        //
        vector<LandmarkObs> transformed_observations;
        for ( vector<LandmarkObs>::iterator observation = observations.begin();
              observation != observations.end(); observation++ ) {

            double sin_theta = sin(particle->theta);
            double cos_theta = cos(particle->theta);

            LandmarkObs transformed_landmark = {
                    observation->id,
                    particle->x + observation->x * cos_theta - observation->y * sin_theta,
                    particle->y + observation->x * sin_theta + observation->y * cos_theta
            };
            transformed_observations.push_back(transformed_landmark);
        }

        //
        // 2. Select landmarks within a reach of particle's sensors
        //
        vector<LandmarkObs> selected_landmarks;
        for ( size_t i = 0; i < map_landmarks.landmark_list.size(); i++ ) {

            Map::single_landmark_s &map_landmark = map_landmarks.landmark_list[i];

            double distance = dist(particle->x, particle->y, map_landmark.x_f, map_landmark.y_f);
            if ( distance <= sensor_range ) {
                LandmarkObs landmark = { map_landmark.id_i, map_landmark.x_f, map_landmark.y_f };
                selected_landmarks.push_back(landmark);
            }
        }

        //
        // 3. Associate selected landmarks with transformed observations
        //
        dataAssociation(selected_landmarks, transformed_observations);

        //
        // 4. compute particle weight
        //
        double weight = 1;

        for (vector<LandmarkObs>::iterator observation = transformed_observations.begin();
                observation != transformed_observations.end(); observation++ ) {

            double dist_x = observation->x - selected_landmarks[observation->id].x;
            double dist_y = observation->y - selected_landmarks[observation->id].y;
            weight *= exp( -( ( dist_x * dist_x / x_denominator ) + ( dist_y * dist_y / y_denominator ) ) )
                      / common_denominator;
        }

        particle->weight = weight;
        weights.push_back(weight);
    }
}

/**
 * Resample particles with replacement with probability proportional to their weight.
 */
void ParticleFilter::resample()
{
    default_random_engine gen;
    // http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::discrete_distribution<int> sample_dist(weights.begin(), weights.end());
    std::vector<Particle> resampled(num_particles);

    for( int i = 0; i < num_particles; i++ ) {
        // sample index
        int sample_idx = sample_dist(gen);
        // save sampled particle to the new list
        resampled[i] = particles[sample_idx];
    }

    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// normal distribution for particles around initial position
	// Set standard deviations for x, y, and theta
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	// creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	num_particles = 100;	// set the number of particles
	// Initialize all particles and weights by normal distribution
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0; //initialize with a weight 1.0
		
		particles.push_back(particle);
		weights.push_back(particle.weight); //initialize the weights vector
	}
	
	// filter is initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// normal distribution for predicting's uncertainty 
	// Set standard deviations for x, y, and theta
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	// creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	// Calculate new state using bicycle model.
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++) {
		Particle &particle = particles[i]; //create a pointer to the ith particle
		
		// bicycle model
		if (fabs(yaw_rate) == 0) 
		{
			particle.x += velocity*delta_t*cos(particle.theta) + dist_x(gen);
			particle.y += velocity*delta_t*sin(particle.theta) + dist_y(gen);
			particle.theta += dist_theta(gen);
		}
		else 
		{
			particle.x += (velocity/yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta)) + dist_x(gen);
			particle.y += (velocity/yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t)) + dist_y(gen);
			particle.theta += yaw_rate*delta_t + dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	// associate observations to the nearest predicted landmarks
	for (unsigned int i = 0; i < observations.size(); i++) { 
		// Initialize min distance as a really big number.
		double minDistance = numeric_limits<double>::max();
		// Initialize the found map in something not possible.
		int mapId = -1;

		// Find the predicted measurement that is closest to each observed measurement
		for (unsigned j = 0; j < predicted.size(); j++ ) { 
			// calculate the distance between observation and predict
			double xDistance = observations[i].x - predicted[j].x;
			double yDistance = observations[i].y - predicted[j].y;
			double distance = xDistance * xDistance + yDistance * yDistance;

			// If the "distance" is less than min, stored the id and update min.
			if ( distance < minDistance ) {
				minDistance = distance;
				mapId = predicted[j].id;
			}
		}
		
		// assign the observed measurement to this particular landmark.
		observations[i].id = mapId;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double stdLandmarkX = std_landmark[0];
	double stdLandmarkY = std_landmark[1];
	
	// update the weights of each particle
	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Transform observation from car coordinates to map coordinates.
		vector<LandmarkObs> mappedObservations;
		for(unsigned int j = 0; j < observations.size(); j++) {
			double xx = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
			double yy = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
			mappedObservations.push_back(LandmarkObs{ observations[j].id, xx, yy });
		}
		
		// Find landmarks in particle's range.
		double sensor_range_2 = sensor_range * sensor_range;
		vector<LandmarkObs> inRangeLandmarks;
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmarkX = map_landmarks.landmark_list[j].x_f;
			float landmarkY = map_landmarks.landmark_list[j].y_f;
			int id = map_landmarks.landmark_list[j].id_i;
			double dX = x - landmarkX;
			double dY = y - landmarkY;

			if ( dX*dX + dY*dY <= sensor_range_2 ) {
				inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
			}
		}
		// Observation association to landmark.
		dataAssociation(inRangeLandmarks, mappedObservations);

		// Reseting weight.
		particles[i].weight = 1.0;
		// Calculate weights.
		for(unsigned int j = 0; j < mappedObservations.size(); j++) {
			// get associated landmark's X,Y
			unsigned int k = 0;
			bool found = false;
			double landmarkX, landmarkY;
			while( !found && k < inRangeLandmarks.size() ) {
				if ( inRangeLandmarks[k].id == mappedObservations[j].id) {
					found = true;
					landmarkX = inRangeLandmarks[k].x;
					landmarkY = inRangeLandmarks[k].y;
				}
				k++;
			}

			// Calculating weight.
			double dX = mappedObservations[j].x - landmarkX;
			double dY = mappedObservations[j].y - landmarkY;
			double weight = ( 1/(2*M_PI*stdLandmarkX*stdLandmarkY)) * exp( -( dX*dX/(2*stdLandmarkX*stdLandmarkX) + (dY*dY/(2*stdLandmarkY*stdLandmarkY)) ) );
			
			particles[i].weight *= weight;
			weights[i] = weight;
		}
	}

	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Take a discrete distribution with pmf equal to weights
    discrete_distribution<> weights_pmf(weights.begin(), weights.end());
	
    // resample particles by discrete distribution
    vector<Particle> newParticles;
	default_random_engine gen;
    for (int i = 0; i < num_particles; ++i){
		newParticles.push_back(particles[weights_pmf(gen)]);
	}
	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
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

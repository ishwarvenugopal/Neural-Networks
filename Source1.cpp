#include <Aria.h>
#include <stdio.h>
#include <iostream>
#include<conio.h>
#include<fstream>
#include<string>
#include<sstream>

using namespace std;

int main(int argc, char **argv)
{
	int i, j;
	//========Parameters for the neural network===============//
	const int no_input = 2, no_output = 2, no_hneuron = 3;
	const double eta = 0.9, lambda = 0.5, alpha = 0.6;
	//=======================================================//
	double wh[no_hneuron][no_input], w[no_output][no_hneuron]; //hidden weights and output weights
	vector<double> row; //to read each row from the file
	string line, word; //to reach the each line and word
	double input[no_input], v[no_hneuron], h[no_hneuron], y[no_output]; //input, output, hidden neurons and net input variables
	double lms_speed, rms_speed; //final speeds
	
	ifstream file_h("finalhiddenweights.csv"); //reading hidden weights
	getline(file_h, line); //reading each line as a string
	while (getline(file_h, line))
	{
		row.clear();
		stringstream ss(line); //for breaking each line into words
		while (getline(ss, word, ','))
		{
			row.push_back(stod(word));
		}
		i = row[0];
		j = row[1];
		wh[i][j] = row[2]; //saving the hidden weights
	}

	ifstream file_o("finaloutputweights.csv"); //reading output weights
	getline(file_o, line); //reading each line as a string
	while (getline(file_o, line))
	{
		row.clear();
		stringstream ss(line); //for breaking each line into words
		while (getline(ss, word, ','))
		{
			row.push_back(stod(word));
		}
		i = row[0];
		j = row[1];
		w[i][j] = row[2]; //saving output weights
	}

	//==========Initialization of robot===============//
	Aria::init();
	ArRobot robot;
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		cout << "Robot Connected!" << endl;
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();
	ArSensorReading *sonarSensor[8];
	int sonarRange[8];
	//===============================================//
	while (true)
	{
		//getting sonar readings
		for (i = 0; i < 8; i++) {
			sonarSensor[i] = robot.getSonarReading(i);
			sonarRange[i] = sonarSensor[i]->getRange();
		}
		//saving normalized inputs
		input[0] = double(min(sonarRange[2],sonarRange[3]))/double(5000); //left front sensor
		input[1] = double(min(sonarRange[0],sonarRange[1]))/double(5000); //left back sensor
		//cout << "Inputs: " << input[0] << " " << input[1] << endl;
		
		for(i=0;i<no_hneuron;i++)
		{
			v[i] = 0;
			for(j=0;j<no_input;j++)
			{
				v[i] += (wh[i][j])*input[j]; //Calculating net input
			}
			h[i]= 1 /( 1 + exp(-lambda*v[i])); //Applying activation function for the hidden neurons
		}
		for (i = 0;i<no_output;i++)
		{
			v[i] = 0;
			for (j = 0;j<no_hneuron;j++)
			{
				v[i] += (w[i][j])*h[j]; //Calculating net input value
			}
			y[i] = 1 /( 1 + exp(-lambda*v[i])); //Applying activation function for output neuron
		}
		//De-normalizing the outputs
		lms_speed = (y[0])*250; 
		rms_speed = (y[1])*250;
		robot.setVel2(lms_speed, rms_speed); //setting robot speeds as per the output obtained
		
		cout << " Output speeds: " << rms_speed << " " << lms_speed << endl;
			
	}

	// Stopping the robot
	robot.lock();
	robot.stop();
	robot.unlock();
	// terminate all threads and exit
	Aria::exit();
	return 0;
}
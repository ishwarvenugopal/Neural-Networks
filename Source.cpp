#include <Aria.h>
#include <stdio.h>
#include <iostream>
#include<conio.h>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
#include<fstream>
#include<string>
#include<sstream>

using namespace std;

//Setting the parameters of the neural network
const int no_input = 2, no_output = 2, no_hneuron = 3, no_epochs = 1000; 
const double eta = 0.9, lambda = 0.3, alpha = 0.6;

class neuron
{
public:
	double value; //to store the value of a neuron
	double error; //to calculate the error for each predicted output
	double wh[no_input], w[no_hneuron]; //the hidden weights and the output weights
	double delta_wh[no_input], delta_w[no_hneuron]; //delta weights
	double delta_wh_old[no_input], delta_w_old[no_hneuron]; //to save the delta weights of previous time step
	double lgrad_hid, lgrad_out; //local gradients
	void initialize_weights(int layer) //function to initialize all the weights, delta weights, local gradients and errors
	{
		int i,j;
		for (j = 0;j < no_input;j++)
		{
			if (layer == 2)
				wh[j] = (double(rand()) / double(RAND_MAX)); //Random number between 0 and 1
			else
				wh[j] = 0;
			delta_wh[j] = double(0);
			delta_wh_old[j] = double(0);
		}
		for (j = 0;j < no_hneuron;j++)
		{
			if (layer == 3)
				w[j] = double(rand()) / double(RAND_MAX); //Random number between 0 and 1
			else
				w[j] = 0;
			delta_w[j] = double(0);
			delta_w_old[j] = double(0);
		}
		lgrad_hid = double(0);
		lgrad_out = double(0);
		error = double(0);
	}
	double activation(string func,double netinput) //function to return a value after applying the activation function
	{
		double activated;
		if (func == "sigmoid")
			activated = 1 / (1 + exp(-lambda*netinput));
		else
			activated = netinput;
		return activated;
	}
};
vector<vector<double>> read_training_data(string filename) //function to read the training data from the csv file
{
	vector<double> row;
	vector <vector<double>> alldata;
	ifstream file(filename);
	string line, word;
	getline(file, line); //reading each line as a string
	while (getline(file, line))
	{
		row.clear();
		stringstream ss(line); //for breaking each line into words
		while (getline(ss, word, ','))
		{
			row.push_back(stod(word));
		}
		alldata.push_back(row);
	}
	return alldata;
}

void main()
{
	vector<vector<double>> training; //to save all the training data
	vector<vector<double>> input, output; //to save the input and output values in the training data
	vector<vector<double>> validation; //to save all the validation data
	vector<vector<double>> v_input, v_output; //to save the input and output values in the validation data
	vector<vector<double>> test; //to save the training data
	vector<vector<double>> test_input, test_output; //to save the input and output values on the testing data
	int i, j, r, k;
	double sum,train_error,val_error,test_error;
	int epoch, total_rows,val_total_rows,test_total_rows;
	vector<vector<double>> wh_avg, w_avg; //Variables to store the average hidden and output weights in each epoch
	srand(time(0)); //initializing a seed (with the system time) to generate a random number

	epoch = 0;
	ofstream errorfile_train("training_errors.csv"); //creating a file to save the training errors
	ofstream errorfile_val("validation_errors.csv"); //creating a file to save the validation errors
	ofstream errorfile_test("test_errors.csv"); //creating a file to save the testing errors
	errorfile_train << "No. of Epochs,Training Error" << endl; //the first row with the column names
	errorfile_val << "No. of Epochs,Validation Error" << endl; //the first row with the column names
	errorfile_test << "Final Average Error" << endl; //the first row with the column names

	ofstream h_weightfile("finalhiddenweights.csv");//file to save the final hidden weights
	ofstream o_weightfile("finaloutputweights.csv");//file to save the final output weights
	h_weightfile << "Hidden Neuron Number,Input neuron number,Weight Value" << endl;
	o_weightfile << "Output Neuron Number,Hidden neuron number,Weight Value" << endl;
	
	training = read_training_data("finaltrainingdata.csv"); //reading all the training data
	for(i=0;i<training.size();i++) //a loop to save the inputs and outputs separately from the training data
	{
		input.push_back(vector<double>());
		input[i].push_back(training[i][0]);
		input[i].push_back(training[i][1]);
		output.push_back(vector<double>());
		output[i].push_back(training[i][2]);
		output[i].push_back(training[i][3]);
	}
	total_rows = training.size(); //determining the total number of rows in the training data

	validation = read_training_data("finalvalidationdata.csv"); //reading all the validation data
	for (i = 0;i<validation.size();i++) //a loop to save the inputs and outputs separately from the validation data
	{
		v_input.push_back(vector<double>());
		v_input[i].push_back(validation[i][0]);
		v_input[i].push_back(validation[i][1]);
		v_output.push_back(vector<double>());
		v_output[i].push_back(validation[i][2]);
		v_output[i].push_back(validation[i][3]);
	}
	val_total_rows = validation.size(); //determining the total number of rows in the validation data

	test = read_training_data("finaltestdata.csv"); //reading all the validation data
	for (i = 0;i<test.size();i++) //a loop to save the inputs and outputs separately from the validation data
	{
		test_input.push_back(vector<double>());
		test_input[i].push_back(test[i][0]);
		test_input[i].push_back(test[i][1]);
		test_output.push_back(vector<double>());
		test_output[i].push_back(test[i][2]);
		test_output[i].push_back(test[i][3]);
	}
	test_total_rows = test.size(); //determining the total number of rows in the validation data
	
	vector<vector<neuron>> hidden; //hidden neurons
	vector<vector<neuron>> predicted; //neurons for predicted outputs
	vector<vector<neuron>> v_hidden; //hidden neurons for validation data
	vector<vector<neuron>> v_predicted; //predicted outputs for validation data
	vector<vector<neuron>> test_hidden; //hidden neurons for validation data
	vector<vector<neuron>> test_predicted; //predicted outputs for validation data

	for (i = 0;i < total_rows;i++) //a loop to initialize each hidden neuron and predicted output neuron
	{
		hidden.push_back(vector<neuron>());
		for (j = 0;j < no_hneuron;j++)
		{
			hidden[i].push_back(neuron());
			hidden[i][j].initialize_weights(2);
		}
		predicted.push_back(vector<neuron>());
		for (j = 0;j < no_output;j++)
		{
			predicted[i].push_back(neuron());
			predicted[i][j].initialize_weights(3);
		}
	}
	for (i = 0;i < val_total_rows;i++) //Initializing hidden and predicted output neuron for validation data
	{
		v_hidden.push_back(vector<neuron>());
		for (j = 0;j < no_hneuron;j++)
		{
			v_hidden[i].push_back(neuron());
			v_hidden[i][j].initialize_weights(2);
		}
		v_predicted.push_back(vector<neuron>());
		for (j = 0;j < no_output;j++)
		{
			v_predicted[i].push_back(neuron());
			v_predicted[i][j].initialize_weights(3);
		}
	}
	for (i = 0;i < test_total_rows;i++) //Initializing hidden and predicted output neuron for testing data
	{
		test_hidden.push_back(vector<neuron>());
		for (j = 0;j < no_hneuron;j++)
		{
			test_hidden[i].push_back(neuron());
			test_hidden[i][j].initialize_weights(2);
		}
		test_predicted.push_back(vector<neuron>());
		for (j = 0;j < no_output;j++)
		{
			test_predicted[i].push_back(neuron());
			test_predicted[i][j].initialize_weights(3);
		}
	}
		
	for (r = 0;r < total_rows;r++) //Initialize the vectors for the average weights
	{
		for (i = 0;i<no_hneuron;i++)
		{
			wh_avg.push_back(vector<double>());
			for (j = 0;j<no_input;j++)
				wh_avg[i].push_back(0);
		}
		for(i=0;i<no_output;i++)
		{
			w_avg.push_back(vector<double>());
			for (j = 0;j < no_hneuron;j++)
				w_avg[i].push_back(0);
		}

	}
	while (epoch < no_epochs) //the main loop that loops over each epoch
	{
		for (r = 0;r < total_rows;r++) //Loops over each row in an epoch
		{
			if(r!=0) //conditions to transfer the weight values from each row to the next
			{
				for (k = 0;k < no_hneuron;k++)
					for (i = 0;i < no_input;i++)
						hidden[r][k].wh[i] = hidden[r - 1][k].wh[i];
				for (k = 0;k < no_output;k++)
					for (i = 0;i < no_hneuron;i++)
						predicted[r][k].w[i] = predicted[r-1][k].w[i];
			}
			else if(r==0)
			{
				if(epoch!=0)
				{
					for (k = 0;k < no_hneuron;k++)
						for (i = 0;i < no_input;i++)
							hidden[r][k].wh[i] = hidden[total_rows-1][k].wh[i];
					for (k = 0;k < no_output;k++)
						for (i = 0;i < no_hneuron;i++)
							predicted[r][k].w[i] = predicted[total_rows - 1][k].w[i];
				}
			}
			
			for (k = 0;k < no_hneuron;k++) //a loop to find the value of each hidden neuron
			{
				sum = 0;
				for (i = 0;i < no_input;i++)
				{
					sum = sum + (hidden[r][k].wh[i] * input[r][i]); //net input
				}
				hidden[r][k].value = hidden[r][k].activation("sigmoid", sum); //applying activation function
			}
			for (k = 0;k < no_output;k++) //a loop to find the value of each predicted output
			{
				sum = 0;
				for (i = 0;i < no_hneuron;i++) 
				{
					sum = sum + (predicted[r][k].w[i] * hidden[r][i].value); //net input
				}
				predicted[r][k].value = predicted[r][k].activation("sigmoid", sum); //applying activation function
			}
			for (i = 0;i<no_output;i++) //a loop to determine the errors in output and their local gradient
			{
				predicted[r][i].error = output[r][i] - predicted[r][i].value;
				predicted[r][i].lgrad_out = lambda*predicted[r][i].value * (1 - predicted[r][i].value)*predicted[r][i].error;
			}
			sum = 0;
			for (i = 0;i<no_hneuron;i++) //finding the local gradient of each hidden neuron
			{
				for (j = 0;j<no_output;j++)
				{
					sum += predicted[r][j].lgrad_out* predicted[r][j].w[i];
				}
				hidden[r][i].lgrad_hid = lambda*hidden[r][i].value * (1 - hidden[r][i].value)*sum;
			}
			for (i = 0;i < no_output;i++) //a loop to calculate the delta weights for output neurons
			{
				for (j = 0;j < no_hneuron;j++)
				{
					if(r!=0)
					{
						predicted[r][i].delta_w_old[j] = predicted[r-1][i].delta_w[j];
						predicted[r][i].delta_w[j] = (eta*predicted[r][i].lgrad_out * hidden[r][j].value) + (alpha*predicted[r][i].delta_w_old[j]);
					}
					if(r==0)
					{
						if(epoch!=0)
						{
							predicted[r][i].delta_w_old[j] = predicted[total_rows-1][i].delta_w[j];
							predicted[r][i].delta_w[j] = (eta*predicted[r][i].lgrad_out * hidden[r][j].value) + (alpha*predicted[r][i].delta_w_old[j]);
						}
					}	
				}
			}
			for (i = 0;i < no_hneuron;i++) //a loop to calculate the delta weights for the hidden neurons
			{
				for (j = 0;j < no_input;j++)
				{
					if (r != 0)
					{
						hidden[r][i].delta_wh_old[j] = hidden[r - 1][i].delta_wh[j];
						hidden[r][i].delta_wh[j] = (eta*hidden[r][i].lgrad_hid* input[r][j]) + (alpha*hidden[r][i].delta_wh_old[j]);
					}
					if (r == 0)
					{
						if (epoch != 0)
						{
							hidden[r][i].delta_wh_old[j] = hidden[total_rows - 1][i].delta_wh[j];
							hidden[r][i].delta_wh[j] = (eta*hidden[r][i].lgrad_hid* input[r][j]) + (alpha*hidden[r][i].delta_wh_old[j]);
						}
					}	
				}
			}
			for (i = 0;i < no_hneuron;i++) //Updating the weights for the hidden neurons
				for (j = 0;j < no_input;j++)
				{
					hidden[r][i].wh[j] = hidden[r][i].wh[j] + hidden[r][i].delta_wh[j];
				}
			for (i = 0;i < no_output;i++) //Updating the weights of the output neurons
				for (j = 0;j < no_hneuron;j++)
					predicted[r][i].w[j] = predicted[r][i].w[j] + predicted[r][i].delta_w[j];
		}

		train_error = 0;
		for (r = 0;r < total_rows;r++) //Calculating the total average training error
		{
			sum = 0;
			for (i = 0;i < no_output ; i++)
			{
				sum += (predicted[r][i].error)*(predicted[r][i].error);
			}
			sum = sum / double(no_output);
			sum = sqrt(sum);
			train_error += sum;
		}
		train_error = train_error / double(total_rows); //Average Training error

		//=======Storing the final epoch weights for the validation data========//
		for (i = 0;i<no_hneuron;i++)
			for (j = 0;j<no_input;j++)
				wh_avg[i][j] = hidden[total_rows-1][i].wh[j];
		for (i = 0;i<no_output;i++)
			for (j = 0;j<no_hneuron;j++)
				w_avg[i][j] = predicted[total_rows-1][i].w[j];
		
		
		//===========Validation Data ===============//
		for(r=0;r<val_total_rows;r++)
		{
			for (k = 0;k < no_hneuron;k++) //a loop to find the value of each hidden neuron
			{
				sum = 0;
				for (i = 0;i < no_input;i++)
				{
					sum = sum + (wh_avg[k][i] * v_input[r][i]); //net input
				}
				v_hidden[r][k].value = v_hidden[r][k].activation("sigmoid", sum); //applying activation function
			}
			for (k = 0;k < no_output;k++) //a loop to find the value of each predicted output
			{
				sum = 0;
				for (i = 0;i < no_hneuron;i++)
				{
					sum = sum + (w_avg[k][i] * v_hidden[r][i].value); //net input
				}
				v_predicted[r][k].value = v_predicted[r][k].activation("sigmoid", sum); //applying activation function
			}
			for (i = 0;i<no_output;i++) //a loop to determine the errors in output
			{
				v_predicted[r][i].error = v_output[r][i] - v_predicted[r][i].value;
			}
		}
		val_error = 0;
		for (r = 0;r < val_total_rows;r++) //Calculating the total average epoch error
		{
			sum = 0;
			for (i = 0;i < no_output; i++)
			{
				sum += (v_predicted[r][i].error)*(v_predicted[r][i].error);
			}
			sum = sum / double(no_output);
			sum = sqrt(sum);
			val_error += sum;
		}
		val_error = val_error / double(val_total_rows); //Average Validation error

		epoch += 1; //incrementing the epoch number
		cout << "Epoch Number: " << epoch << " , Training Error: " << train_error << " , Validation Error: " << val_error << endl; //Displaying the epoch errors on the screen
		errorfile_train << epoch << "," << train_error << endl; //Writing the training errors to a csv file
		errorfile_val << epoch << "," << val_error << endl; //Writing the validation errors to a csv file
	} //The training and validation session ends here
	
	
	//=======Storing the final epoch weights for the testing data========//
	for (i = 0;i<no_hneuron;i++)
		for (j = 0;j<no_input;j++)
		{
			wh_avg[i][j] = hidden[total_rows - 1][i].wh[j];
			h_weightfile << i << "," << j << "," << wh_avg[i][j] << endl;
		}
			
	for (i = 0;i<no_output;i++)
		for (j = 0;j<no_hneuron;j++)
		{
			w_avg[i][j] = predicted[total_rows - 1][i].w[j];
			o_weightfile << i << "," << j << "," << w_avg[i][j] << endl;
		}

	//===========Testing Data ===============//
	for (r = 0;r<test_total_rows;r++)
	{
		for (k = 0;k < no_hneuron;k++) //a loop to find the value of each hidden neuron
		{
			sum = 0;
			for (i = 0;i < no_input;i++)
			{
				sum = sum + (wh_avg[k][i] * test_input[r][i]); //net input
			}
			test_hidden[r][k].value = test_hidden[r][k].activation("sigmoid", sum); //applying activation function
		}
		for (k = 0;k < no_output;k++) //a loop to find the value of each predicted output
		{
			sum = 0;
			for (i = 0;i < no_hneuron;i++)
			{
				sum = sum + (w_avg[k][i] * test_hidden[r][i].value); //net input
			}
			test_predicted[r][k].value = test_predicted[r][k].activation("sigmoid", sum); //applying activation function
		}
		for (i = 0;i<no_output;i++) //a loop to determine the errors in output
		{
			test_predicted[r][i].error = test_output[r][i] - test_predicted[r][i].value;
		}
	}
	test_error = 0;
	for (r = 0;r < test_total_rows;r++) //Calculating the total average epoch error
	{
		sum = 0;
		for (i = 0;i < no_output; i++)
		{
			sum += (test_predicted[r][i].error)*(test_predicted[r][i].error);
		}
		sum = sum / double(no_output);
		sum = sqrt(sum);
		test_error += sum;
	}
	test_error = test_error / double(test_total_rows); //Average Validation error
	cout << "Average Testing Error: " << test_error << endl;
	errorfile_test << test_error << endl;
	getch();
}
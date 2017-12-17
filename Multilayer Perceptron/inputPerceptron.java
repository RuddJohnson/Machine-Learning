/**
 * Rudd Johnson
 * 4/24/17
 */
import java.math.*;
import  java.util.Random;

public class inputPerceptron {                      //class for input layer perceptron
    private double output;                          //store output
    private double [] weight;                       //array of weights
    private double n;                               //learning rate
    private double weightInputSum;                  //dot product weights and inputs
    private double momentum;                        //hold momentum
    private double delta;                           //hold error rate
    private double [] weightChange;                 //array of changed weights

    public inputPerceptron(double n, double momentum) {         //constructor
        this.n = n;
        this.weight = new double[785];                          //create array of weights
        for(int i=0;i<785;++i){weight[i] = rng();}              //random initial weights between -0.05 - 0.05

        this.momentum = momentum;
        this.delta=0;
        this.weightInputSum =0;
        this.weightChange = new double[785];                    //array of weight change equal in size to weight
        for(int i=0;i<weight.length;++i){                       //initialize to 0
            this.weightChange[i] = 0.0;
        }
    }

    public double activation(double [] row){                    //dot product of input and input weights, stored in weighInputSum
        this.weightInputSum = 0.0;                              //bias from hidden layer
        double temp = row[0];
        row[0] = 1.0;                                           //store input cal in temp and set val to 1 for bias input
        for(int i =0; i< row.length; ++i) {this.weightInputSum += row[i] * weight[i];}  //perform dot product
        row[0] = temp;

        this.weightInputSum = sigmoid();                         //run sigmoud on dot product and return result for use as hidden layer input
        return weightInputSum;
    }
    //return result of sigmoid function to calling routine, is used as input for hidden layer
    public double sigmoid(){return (double)(1/(1 + (Math.exp(-weightInputSum))));}

    public void adjustDelta(double [] outputDelta, double [] outputWeights){                    //find the error rate of input perceptron
        double sum =0.0;
        for(int i=0; i<outputDelta.length;++i){sum += (outputDelta[i] * outputWeights[i]);}     //sum error rate of ouput layer by corresponding weights to that perceptron
        this.delta = (weightInputSum) * (1-weightInputSum)*(sum);                               //multiply by 1-sigmoid time sigmoid and store in delta
    }

    public void adjustWeight(double [] row, double [] outputDelta, double [] outputWeight){     //adjust the weight of input perceptron
        adjustDelta(outputDelta,outputWeight);                                                  //first adjust the delta
        double inputVal = row[0];                                                               //store input value in temp and set that place in row to bias input
        row[0] = 1.0;

        for(int i=0;i<weight.length; ++i) {                                                     //iterate through all weights
            double temp = weight[i];                                                            //hold previous weight
            weight[i] = weight[i] + (n * delta * row[i]) + (momentum * weightChange[i]);        //calculate new weight
            weightChange[i] = weight[i] - temp;                                                 //find weight change by subtracting old weight from new weight
        }
        row[0] = inputVal;                                                                      //reset row index to correct value

    }

    public double rng(){                                                                        //random number generator for setting initial weights to
        Random rand = new Random();                                                             //between -0.05-0.05
        double scale = (0.1 + rand.nextDouble() *0.01);
       return ((rand.nextInt(10) -4) * scale)*.1;
    }
}

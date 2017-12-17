/*
 * Rudd Johnson
 * 4/24/17
 */
import java.math.*;
import  java.util.Random;

public class hiddenLayer {                  //class for hidden layer perceptron
    private double classification;          //store classification
    private double input;                   //store input
    private double [] weight;               //store array of weights
    private double n;                       //learing rate
    private double weightInputSum;          //dot product of weights and inputs
    private double momentum;                //momentum
    private double delta;                   //store error rate
    private double [] weightChange;         //store weight change
    private int numHidden;                  //store num hidden
    private double bias;                    //store bias
    private double biasChange;              //store bias change (weight change)
    public hiddenLayer(double classification, double n, double momentum, int numHidden) {       //constructor
        this.n = n;
        this.numHidden = numHidden;
        this.weight = new double[numHidden];                        //create weights for all hidden layer inputs

        for(int i=0;i<numHidden;++i){this.weight[i] = rng();}       //random initial weights between -0.05 - 0.05
        this.bias = rng();
        this.biasChange =0.0;

        this.momentum = momentum;
        this.delta =0;
        this.weightChange = new double[numHidden];                   //array of weight change equal in size to weight
        for(int i=0;i<numHidden;++i){                                //initialize to 0
            this.weightChange[i] = 0.0;
        }
        this.classification=classification;
    }

    public double activation(double [] inputLayer, double value){       //dot product of input and input weights, stored in weighInputSum
        this.weightInputSum = 1 * bias;                                 //bias input
        double target =0.0;
        for(int i =0; i< inputLayer.length; ++i) {this.weightInputSum += inputLayer[i] * weight[i];}        //perform dot product

        this.weightInputSum = sigmoid();                                  //run sigmoid on dot product

        if(value == classification) {target = 0.9;}                       //classify as 0.9 if input value is the same as perceptron class, else 0.1
        else{target = 0.1;}

        return adjustDelta(target);                                       //return error rate
    }

    public void activationNoWeightAdjust(double [] inputLayer){           //activation for training set where no weights adjusted, redundent, same as activation without
        this.weightInputSum = 1 * bias;                                   //error calculation, identical to testActivation
        for(int i =1; i< inputLayer.length; ++i) {
            this.weightInputSum += inputLayer[i] * weight[i];
        }
        this.weightInputSum = sigmoid();
    }

                                                                        //same as activation except error weight not calculated because
    public void testActivation(double [] inputLayer){                   //weights are not being adjusted
        this.weightInputSum =1 * bias;
        for(int i =0; i< inputLayer.length; ++i) {
            this.weightInputSum += inputLayer[i] * weight[i];
        }
       this.weightInputSum = sigmoid();
    }
    //return result of sigmoid function to calling routine, weightInputSum set to this value
    public double sigmoid(){
        return (double)(1/(1 + (Math.exp(-weightInputSum))));
    }

    public double adjustDelta(double target){                                       //calculate error rate
        this.delta = weightInputSum *(1-weightInputSum)*(target-weightInputSum);    //multiply sigmoid by 1-sigmoid time sigmoid and target (0.1 or 0.9)
        return delta;
    }

    public void adjustWeight(double [] inputLayer){                                 //adjust weight of hiddenLayerr ->output
        double tempBias = bias;                                                     //store bias weight in temp
        this.bias = bias + (n * delta * 1) + (momentum * biasChange);               //calculate new bias weight
        this.biasChange = bias - tempBias;                                          //find difference in new bias weight and old bias weight

        for(int i=0;i<weight.length; ++i) {                                         //iterate through all weights
            double temp = weight[i];                                                //store old weight in temp
            this.weight[i] = weight[i] + (n * delta * inputLayer[i]) + (momentum * weightChange[i]);        //calculate new weight
            this.weightChange[i] = weight[i] - temp;                                //calculate weight change (difference between new weight and old weight)
        }
    }

    public double rng(){                                    //random number generator for setting initial weights to
        Random rand = new Random();                         //between -0.05-0.05
        double scale = (0.1 + rand.nextDouble() *0.01);
        return ((rand.nextInt(10) -4) * scale)*.1;
    }

    public double getWeight(int index){ return weight[index];}      //return specific weight

    public double getSum(){return weightInputSum;}                  //return dot product weight and input

    public double getClassification(){return classification;}       //return perceptron class
}

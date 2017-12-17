/**
 * Rudd Johnson
 * 4/24/17
 * Machine Learning
 * Assignment 2
 */

import java.io.*;

public class Manager {                  //Class manages reading in / creation of test and train file information, hiddenlayer and outer layer creation
    FileInputStream in = null;
    private double[][] testFileData;    //matrix for test file data
    private double[][] trainFileData;   //matrix for train file data
    private inputPerceptron[] inputTohidden;    //holds result from input layer to be input into hidden layer
    private hiddenLayer[] hiddenTooutput;       //holds result from hidden to output layer
    private double[] trainAccuracy;             //store training accuracy
    private double[] testAccuracy;              //store testing accuracy
    private double[] inputResult;               //store input layer result
    private double[] outputError;               //store error from output
    private int numHidden;                      //store size of hidden layer

    Manager(double momentum, double learningRate, int numHidden) {      //constructor for manager
        this.numHidden = numHidden;                                     //set private data members
        this.testAccuracy = new double[50];                             //create arrays and matrices
        this.trainAccuracy = new double[50];
        this.testFileData = new double[10000][785];
        this.trainFileData = new double[60000][785];

        this.inputTohidden = new inputPerceptron[numHidden];
        for (int i = 0; i < numHidden; ++i) {                           //instanciate each perceptron in input
            inputTohidden[i] = new inputPerceptron(learningRate, momentum);
        }

        this.hiddenTooutput = new hiddenLayer[10];                      //instantiate each perceptron in output
        for (int i = 0; i < 10; ++i) {
            hiddenTooutput[i] = new hiddenLayer(((double)i / 255), learningRate, momentum, numHidden);
        }

        this.inputResult = new double[numHidden];                       //initialize result from input layer
        for (int i = 0; i < numHidden; ++i) {
            inputResult[i] = 0.0;
        }

        this.outputError = new double[10];                              //initialize result from output error
        for (int i = 0; i < 10; ++i) {
            outputError[i] = 0.0;
        }

        String row = "";
        try {                                                             //read in training information
            FileReader reader = new FileReader("mnist_train.csv");
            BufferedReader bufferedReader = new BufferedReader(reader);
            int x = 0;
            while ((row = bufferedReader.readLine()) != null) {
                String[] rowSplit = row.split(",");                         //format such that each value is own element in array
                double[] trainConversion = new double[rowSplit.length];
                for (int i = 0; i < rowSplit.length; ++i) {
                    trainConversion[i] = Double.parseDouble(rowSplit[i]);
                    trainConversion[i] = (double)trainConversion[i] / 255;  //divide by 255
                    this.trainFileData[x][i] = trainConversion[i];          //add to matix of file info
                }
                ++x;
            }


            bufferedReader.close();     //read and format test data
            reader = new FileReader("mnist_test.csv");                      //read in test file data
            bufferedReader = new BufferedReader(reader);
            x = 0;
            while ((row = bufferedReader.readLine()) != null) {
                String[] rowSplit = row.split(",");                         //format such that each value is own element in array
                double[] testConversion = new double[rowSplit.length];
                for (int i = 0; i < rowSplit.length; ++i) {
                    testConversion[i] = Double.parseDouble(rowSplit[i]);
                    testConversion[i] = (double)testConversion[i] / 255;    //divide by 255
                    this.testFileData[x][i] = testConversion[i];            //add to matix of file info
                }
                ++x;
            }
            bufferedReader.close();                                          //close file

        } catch (FileNotFoundException excep) {                             //capture exceptions
            excep.printStackTrace();
        } catch (IOException excep) {
            excep.printStackTrace();
        }
    }

    public int train() {                                                    //training function
        int [] vals = new int [10];
        intializeCount(vals);
        int correct = 0;                                                    //correct number of predictions
        double[] hideOutWeights = new double[10];                           //holds weights from hidden layer to output
        double[][] oldWeightMatrix = new double[numHidden][10];             //holds weights from output to hidden for each hidden perceptron
        double highestSum = -1000000.0;                                     //store highest sum
        double predictVal = 0.0;                                            //hold predicted class
        for (int k = 0; k < 60000; ++k) {                                   //iterate over every test valuse
            highestSum = -1000000.0;                                        //reset highest sum and predicted class
            predictVal = 0.0;
         //  if(halfSet(trainFileData[k][0], vals) == false) {continue;}    //if modifying the training set, ensure only half of training set is used with even value distribution

            for (int i = 0; i < numHidden; ++i) {                                   //call activation on input layer
                inputResult[i] = inputTohidden[i].activation(trainFileData[k]);
            }

            for(int i=0;i<10;++i){                                                  //call activation on hidden layer
                outputError[i] = hiddenTooutput[i].activation(inputResult, trainFileData[k][0]);
                getOutputweight(i, hideOutWeights);                                 //get all weights going from hidden layer to outputs
                copyWeights(i, hideOutWeights, oldWeightMatrix);
            }

            for (int i = 0; i < 10; ++i) {                                                      //backprop, first adjust output weights and find output class
                hiddenTooutput[i].adjustWeight(inputResult);
                if (highestSum < hiddenTooutput[i].getSum()) {
                    highestSum = hiddenTooutput[i].getSum();
                    predictVal = hiddenTooutput[i].getClassification();
                }
            }
            if (trainFileData[k][0] == predictVal) {                                           //if output class correct, increment correct
                correct = correct + 1;
            }
            for (int i = 0; i < numHidden; ++i) {                                             //backprop to input layer, adjust weights
                inputTohidden[i].adjustWeight(trainFileData[k], outputError, oldWeightMatrix[i]);
            }
        }
            return correct;                                                                  //return correct predictions
    }


    public int trainNoWeightAdjust(){                                                       //check accuracy of training set
        int [] vals = new int [10];
        intializeCount(vals);

        int correct =0;                                                                     //uses methods described in train but without adjusting weights
        for (int k = 0; k < 60000; ++k) {
            double highestSum = -10000000.0;
            double predictVal = 0.0;
//            if(halfSet(trainFileData[k][0], vals) == false) {continue;}

            for (int i = 0; i < numHidden; ++i) {
                inputResult[i] = inputTohidden[i].activation(trainFileData[k]);
            }

            for (int i = 0; i < 10; ++i) {                                                      //forward prop and begin backprop
                hiddenTooutput[i].activationNoWeightAdjust(inputResult);
                if(highestSum < hiddenTooutput[i].getSum()){
                    highestSum = hiddenTooutput[i].getSum();
                    predictVal = hiddenTooutput[i].getClassification();
                }
            }
            if(trainFileData[k][0] == predictVal){correct = correct+1; }
        }
        return correct;
    }

    public int test(){                                                          //function runs test set through trained network (at each epoch)
        int correct =0;
        double highestSum = -1000000.0;                                        //store highestSum
        double predictVal = 0.0;
        for (int k = 0; k < 10000; ++k) {                                       //iterate over training data
             highestSum = -1000000.0;                                           //reset highest val and predictVal
             predictVal = 0.0;

            for (int i = 0; i < numHidden; ++i) {                               //activate input layer
                inputResult[i] = inputTohidden[i].activation(testFileData[k]);
            }

            for(int i=0;i<10;++i){                                              //activate hidden layer
                hiddenTooutput[i].testActivation(inputResult);                  //find output classification
                 if(highestSum < hiddenTooutput[i].getSum()){
                     highestSum = hiddenTooutput[i].getSum();
                     predictVal = hiddenTooutput[i].getClassification();
                 }
            }
            if(testFileData[k][0] == predictVal){correct = correct+1;}         //if classication correct, increment correct
        }

        return correct;                                                         //return correct
    }

    public void getOutputweight(int index, double[] weightList) {               //store weights in each output layer perceptron in array
        for (int i = 0; i < 10; ++i) {
            weightList[i] = hiddenTooutput[i].getWeight(index);
        }
    }

    public void copyWeights(int index,double [] weights, double [][] weightMatrix){
        for(int i=0; i<weights.length; ++i){                                    //store array of weights in matrix where each index correlates to hidden layer perceptron
            weightMatrix[index][i]= weights[i];
        }
    }

    public void confuseMatrix(int [][] confuse){                                //create confusion matrix
        for (int k = 0; k < 10000; ++k) {
            double highestSum = -1000000.0;                                     //runs test(), but instead of outputtin correct or incorrect, stores results in matrix
            double predictVal = 0.0;
            for (int i = 0; i < numHidden; ++i) {                               //activate input layer
                inputResult[i] = inputTohidden[i].activation(testFileData[k]);
            }

            for(int i=0;i<10;++i){                                              //activate hidden layer
                hiddenTooutput[i].testActivation(inputResult);                  //find predicted class
                if(highestSum < hiddenTooutput[i].getSum()){
                    highestSum = hiddenTooutput[i].getSum();
                    predictVal = hiddenTooutput[i].getClassification();
                }
            }
            int predictValint = (int)(predictVal*255);                          //store predicted class along with actual class in matrix
            int inputClass = (int)(testFileData[k][0]*255);
            ++confuse[inputClass][predictValint];
        }
    }

    public void intializeCount(int [] valCount){                                //initialize array holding count for each input type
        for(int i=0; i<10; ++i){valCount[i] =0;}
    }                                                                           //enforce quarter of training set used
    public boolean quarterSet(double val, int [] valCount){                     //if more than 1500 of any type of number has been  trained on, return false (ensures balance)
            int convert = (int)(val * 255);
            if(valCount[convert] < 1500){
                ++valCount[convert];
                return true;
            }
            return false;
    }
    public boolean halfSet(double val, int [] valCount){                          //enforce half of training set used
        int convert = (int)(val * 255);                                           //if more than 3000 of any type of number has been  trained on, return false (ensures balance)
        if(valCount[convert] < 3000){
            ++valCount[convert];
            return true;
        }
        return false;
    }
}

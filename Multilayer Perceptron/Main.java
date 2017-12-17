/*
 * Rudd Johnson
 * 4/24/17
 */
import java.util.Scanner;
import java.util.Random;
public class Main {                                     //entry point
    public static void main(String[] args) {
        Scanner input=new Scanner(System.in);           //for user I/O
        double [] trainResults = new double[50];        //hold results from training
        double [] testResult = new double[50];          //hold results from testing
        int [][] confusionMatrix = new int[10][10];     //hold confusion matrix result
        double momentum,learningRate;                   //store momentum and learning input
        int numHidden =0;                               //store size hidden layer
        momentum = learningRate = 0.0;

        System.out.println("please input momentum: ");               //input momentum
        momentum = input.nextFloat();
        input.nextLine();

        System.out.println("please input learning rate: ");         //input learning rate
        learningRate = input.nextFloat();
        input.nextLine();

        System.out.println("please input number hidden nodes: ");   //input size hidden layer
        numHidden = input.nextInt();
        input.nextLine();

        Manager learn = new Manager(momentum, learningRate, numHidden);                   //instantiate manager class
         //epoch zero
         trainResults[0] = (((double)(learn.trainNoWeightAdjust())/60000) * 100);         //store accuracy percentage for epoch zero train
         testResult[0] = (((double)(learn.test())/10000) *100);                           //store accuracy percentage for epoch zero test


        for(int i =1; i<50; ++i){                                                        //run for 50 epochs
            learn.train();                                                               //train, adjust weights
            trainResults[i] = (((double) (learn.trainNoWeightAdjust())/60000) * 100);    //store accuracy of training set without adjusting weights
            testResult[i] = (((double)(learn.test())/10000) *100);                       //run test set, store results
        }

        System.out.println("Training: "+numHidden);                                     //output training accuracy
        for(int i =0; i<50; ++i){
            System.out.println(trainResults[i]);
        }

        System.out.println("Testing: "+numHidden);                                      //output testing accuracy
        for(int i =0; i<50; ++i){
            System.out.println(testResult[i]);
        }
        learn.confuseMatrix(confusionMatrix);                                           //create confusion matrix
        for (int i=0; i<10;++i){                                                        //output confusion matrix
            for (int j =0; j<10; ++j) {
                System.out.print(confusionMatrix[i][j]+",");
            }
            System.out.print("\n");
        }

    }
}

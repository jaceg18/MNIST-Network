import Logic.NeuralNetwork;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork();

        int numTrainingExamples = 60000;
        int numTestingExamples = 10000;

        double[][] trainingData = new double[numTrainingExamples][NeuralNetwork.INPUT_LAYER_SIZE];
        int[] trainingLabels = new int[numTrainingExamples];

        double[][] testingData = new double[numTestingExamples][NeuralNetwork.INPUT_LAYER_SIZE];
        int[] testingLabels = new int[numTestingExamples];


        // Load training data
        Scanner sc = new Scanner(new FileInputStream("src/TrainingData/mnist_train.csv"));
        sc.nextLine();
        for (int i = 0; i < numTrainingExamples; i++) {
            String line = sc.nextLine();
            String[] parts = line.split(",");
            trainingLabels[i] = Integer.parseInt(parts[0]);

            for (int j = 0; j < NeuralNetwork.INPUT_LAYER_SIZE; j++)
                trainingData[i][j] = Double.parseDouble(parts[j + 1]) / 255;

        }

        sc.close();

        sc = new Scanner(new FileInputStream("src/TrainingData/mnist_test.csv"));
        sc.nextLine();
        for (int i = 0; i < numTestingExamples; i++) {
            String line = sc.nextLine();
            String[] parts = line.split(",");
            testingLabels[i] = Integer.parseInt(parts[0]);
            for (int j = 0; j < NeuralNetwork.INPUT_LAYER_SIZE; j++)
                testingData[i][j] = Double.parseDouble(parts[j + 1]) / 255;// / 255.0;

        }
        sc.close();

        int epoffs = 8;
        for (int i = 0; i < epoffs; i++) {
            for (int j = 0; j < numTrainingExamples; j++)
                nn.train(trainingData[j], trainingLabels[j]);


            System.out.println((i+1) + " sets trained out of " + epoffs + " sets");
        }


        int numCorrect = 0;
        for (int i = 0; i < numTestingExamples; i++) {
            int prediction = nn.predict(testingData[i]);
            if (prediction == testingLabels[i]) {
                numCorrect++;
            }
        }

        System.out.println("Accuracy: " + (double) numCorrect / numTestingExamples);
        System.out.println(numCorrect + " correct out of " + numTestingExamples);
    }

}
import Logic.NeuralNetwork;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

public class Main2 {
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork();

        int numTrainingExamples = 60000;
        int numValidationExamples = 10000;
        int numTestingExamples = 10000;

        double[][] trainingData = new double[numTrainingExamples][NeuralNetwork.INPUT_LAYER_SIZE];
        int[] trainingLabels = new int[numTrainingExamples];

        double[][] validationData = new double[numValidationExamples][NeuralNetwork.INPUT_LAYER_SIZE];
        int[] validationLabels = new int[numValidationExamples];

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
        for (int i = 0; i < numValidationExamples; i++) {
            String line = sc.nextLine();
            String[] parts = line.split(",");
            validationLabels[i] = Integer.parseInt(parts[0]);
            for (int j = 0; j < NeuralNetwork.INPUT_LAYER_SIZE; j++)
                validationData[i][j] = Double.parseDouble(parts[j + 1]) / 255;// / 255.0;

        }
        sc.close();
        sc = new Scanner(new FileInputStream("src/TrainingData/mnist_test.csv"));
        sc.nextLine();
        for (int i = numValidationExamples; i < numValidationExamples + numTestingExamples; i++) {
            String line = sc.nextLine();
            String[] parts = line.split(",");
            testingLabels[i - numValidationExamples] = Integer.parseInt(parts[0]);
            for (int j = 0; j < NeuralNetwork.INPUT_LAYER_SIZE; j++)
                testingData[i - numValidationExamples][j] = Double.parseDouble(parts[j + 1]) / 255;// / 255.0;

        }
        sc.close();

        int epochs = 10;
        int patience = 5;
        int bestEpoch = -1;
        double bestAccuracy = 0;
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < numTrainingExamples; j++)
             nn.train(trainingData[j], trainingLabels[j]);

            double accuracy = nn.evaluate(validationData, validationLabels);
            System.out.println("Epoch " + (i + 1) + " accuracy: " + accuracy);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestEpoch = i;
            } else if (i - bestEpoch >= patience) {
                break;
            }
        }
        System.out.println("Best epoch: " + (bestEpoch + 1) + " with accuracy: " + bestAccuracy);
        System.out.println("Final test accuracy: " + nn.evaluate(testingData, testingLabels));
    }
}


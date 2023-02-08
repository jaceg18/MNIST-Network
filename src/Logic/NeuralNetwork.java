package Logic;
import java.util.Random;

public class NeuralNetwork {

    public static final int INPUT_LAYER_SIZE = 784;
    public static final int HIDDEN_LAYER_SIZE = 100;
    public static final int OUTPUT_LAYER_SIZE = 10;
    public static final double LEARNING_RATE = 0.035; // average 90% accuracy

    private double[][] inputToHiddenWeights;
    private double[] hiddenBias;
    private double[][] hiddenToOutputWeights;
    private double[] outputBias;

    public NeuralNetwork() {
        inputToHiddenWeights = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
        hiddenBias = new double[HIDDEN_LAYER_SIZE];
        hiddenToOutputWeights = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];
        outputBias = new double[OUTPUT_LAYER_SIZE];

        Random rng = new Random();

        for (int i = 0; i < INPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                inputToHiddenWeights[i][j] = rng.nextDouble() * 0.1; // * 0.1
            }
        }

        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            hiddenBias[i] = rng.nextDouble() * 0.1;
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                hiddenToOutputWeights[i][j] = rng.nextDouble() * 0.1;
            }
        }

        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            outputBias[i] = rng.nextDouble() * 0.1;
        }
    }

    public int predict(double[] inputs) {
        double[] hiddenLayerActivations = new double[HIDDEN_LAYER_SIZE];

        // Calculate hidden layer activations
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < INPUT_LAYER_SIZE; j++) {
                hiddenLayerActivations[i] += inputToHiddenWeights[j][i] * inputs[j];
            }
            hiddenLayerActivations[i] += hiddenBias[i];
            hiddenLayerActivations[i] = sigmoid(hiddenLayerActivations[i]);
        }

        double[] outputLayerActivations = new double[OUTPUT_LAYER_SIZE];

        // Calculate output layer activations
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                outputLayerActivations[i] += hiddenToOutputWeights[j][i] * hiddenLayerActivations[j];
            }
            outputLayerActivations[i] += outputBias[i];
            outputLayerActivations[i] = sigmoid(outputLayerActivations[i]);
        }

        int maxIndex = 0;
        for (int i = 1; i < OUTPUT_LAYER_SIZE; i++) {
            if (outputLayerActivations[i] > outputLayerActivations[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public void train(double[] inputs, int target) {
        double[] hiddenLayerActivations = new double[HIDDEN_LAYER_SIZE];

        // Calculate hidden layer activations
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < INPUT_LAYER_SIZE; j++)
                hiddenLayerActivations[i] += inputToHiddenWeights[j][i] * inputs[j];

            hiddenLayerActivations[i] += hiddenBias[i];
            hiddenLayerActivations[i] = sigmoid(hiddenLayerActivations[i]);
        }

        double[] outputLayerActivations = new double[OUTPUT_LAYER_SIZE];

        // Calculate output layer activations
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
                outputLayerActivations[i] += hiddenToOutputWeights[j][i] * hiddenLayerActivations[j];

            outputLayerActivations[i] += outputBias[i];
            outputLayerActivations[i] = sigmoid(outputLayerActivations[i]);
        }

        double[] outputLayerErrors = new double[OUTPUT_LAYER_SIZE];

        // Calculate output layer errors
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
            outputLayerErrors[i] = outputLayerActivations[i] - (i == target ? 1 : 0);


        double[] hiddenLayerErrors = new double[HIDDEN_LAYER_SIZE];

        // Calculate hidden layer errors
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++)
                hiddenLayerErrors[i] += outputLayerErrors[j] * hiddenToOutputWeights[i][j];



        // Update hidden to output weights
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++)
                hiddenToOutputWeights[i][j] -= LEARNING_RATE * outputLayerErrors[j] * hiddenLayerActivations[i];



        // Update output biases
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            outputBias[i] -= LEARNING_RATE * outputLayerErrors[i];
        }

        // Update input to hidden weights
        for (int j = 0; j < INPUT_LAYER_SIZE; j++)
            for (int k = 0; k < HIDDEN_LAYER_SIZE; k++)
                inputToHiddenWeights[j][k] -= LEARNING_RATE * hiddenLayerErrors[k] * inputs[j];



        // Update hidden biases
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            hiddenBias[i] -= LEARNING_RATE * hiddenLayerErrors[i];
        }
    }
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double evaluate(double[][] data, int[] labels) {
        int correct = 0;
        for (int i = 0; i < data.length; i++) {
            int predicted = predict(data[i]);
            if (predicted == labels[i]) {
                correct++;
            }
        }
        return (double) correct / data.length;
    }

}

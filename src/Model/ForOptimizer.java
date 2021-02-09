package Model;

import java.util.ArrayList;
import java.util.function.Function;

public class ForOptimizer {
    public Model[] Training;
    public int BatchSize;
    public Function<ForLoss, Double> Loss;
    public ArrayList<Integer> Ytrain;
    public double learningRate;

    public ForOptimizer(Model[] Training, ArrayList<Integer> Ytrain, int BatchSize, Function<ForLoss, Double> Loss, double learningRate) {
        this.Training = Training;
        this.BatchSize = BatchSize;
        this.Loss = Loss;
        this.Ytrain = Ytrain;
        this.learningRate = learningRate;
    }
}

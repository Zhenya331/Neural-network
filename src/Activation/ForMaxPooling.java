package Activation;

public class ForMaxPooling implements IActivation {
    @Override
    public double[] Method(double[] data) {
        return data;
    }

    @Override
    public double[] Derivative(double[] data) {
        return data;
    }
}

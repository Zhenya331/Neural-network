package Activation;

public class Tanh implements IActivation{
    @Override
    public double[] Method(double[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (Math.exp(data[i]) - Math.exp(-data[i])) / (Math.exp(data[i]) + Math.exp(-data[i]));
        }
        return data;
    }

    @Override
    public double[] Derivative(double[] data) {
        double[] bufRes = Method(data);
        for (int i = 0; i < data.length; i++) {
            data[i] = 1 - bufRes[i] * bufRes[i];
        }
        return data;
    }
}

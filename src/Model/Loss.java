package Model;

public class Loss {
    public static double mse(ForLoss arg) {
        double res = 0;
        for(int i = 0; i < arg.Result.length; i++) {
            if (arg.Answer == i) {
                res += Math.pow(1 - arg.Result[i], 2);
                continue;
            }
            res += Math.pow(arg.Result[i], 2);
        }
        return res / 2;
    }
}

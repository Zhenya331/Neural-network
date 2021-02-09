package Model;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Log {
    private static final String pathToDirectory = "C:\\Users\\lorgo\\Desktop\\Study\\Магистратура\\1 курс\\Нейронные сети\\Statistic\\";
    public static ArrayList<Double> trainError = new ArrayList<>();
    public static ArrayList<Double> trainAccuracy = new ArrayList<>();
    public static ArrayList<Double> testError = new ArrayList<>();
    public static ArrayList<Double> testAccuracy = new ArrayList<>();

    public static void GetTrainError(int epoch) {
        String fileName = "TrainError" + epoch + ".txt";
        File f = new File(pathToDirectory + fileName);
        if(!f.exists()){ try { f.createNewFile(); } catch (IOException e) { e.printStackTrace(); } } else { f.delete(); }

        try(FileWriter writer = new FileWriter(f, true))
        {
            for(double trainings: trainError) {
                writer.write(String.valueOf(trainings) + '\n');
            }
            writer.flush();
        }
        catch(IOException ex){
            System.out.println(ex.getMessage());
        }
        trainError = new ArrayList<>();
    }

    public static void GetTrainAccuracy() {
        String fileName = "TrainAccuracy.txt";
        File f = new File(pathToDirectory + fileName);
        if(!f.exists()){ try { f.createNewFile(); } catch (IOException e) { e.printStackTrace(); } } else { f.delete(); }

        try(FileWriter writer = new FileWriter(f, true))
        {
            for(double trainAcc: trainAccuracy) {
                writer.write(String.valueOf(trainAcc) + '\n');
            }
            writer.flush();
        }
        catch(IOException ex){
            System.out.println(ex.getMessage());
        }
        trainAccuracy = new ArrayList<>();
    }

    public static void GetTestError() {
        String fileName = "TestError.txt";
        File f = new File(pathToDirectory + fileName);
        if(!f.exists()){ try { f.createNewFile(); } catch (IOException e) { e.printStackTrace(); } } else { f.delete(); }

        try(FileWriter writer = new FileWriter(f, true))
        {
            for(double testErr: testError) {
                writer.write(String.valueOf(testErr) + '\n');
            }
            writer.flush();
        }
        catch(IOException ex){
            System.out.println(ex.getMessage());
        }
        testError = new ArrayList<>();
    }

    public static void GetTestAccuracy() {
        String fileName = "TestAccuracy.txt";
        File f = new File(pathToDirectory + fileName);
        if(!f.exists()){ try { f.createNewFile(); } catch (IOException e) { e.printStackTrace(); } } else { f.delete(); }

        try(FileWriter writer = new FileWriter(f, true))
        {
            for(double testAcc: testAccuracy) {
                writer.write(String.valueOf(testAcc) + '\n');
            }
            writer.flush();
        }
        catch(IOException ex){
            System.out.println(ex.getMessage());
        }
        testAccuracy = new ArrayList<>();
    }
}

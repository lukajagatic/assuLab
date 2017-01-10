package assu_lab;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class NetRunner {

	public static void main(String[] args) throws Exception {
		int input = 0, hidden = 0, output = 0, cycles = 0, numberOfPatterns = 0;
		String patternFile = null, outputFile = null, testFile = null;
		double learningRate = 0, moment = 0;
		for (int i = 0; i < args.length; i++) {
			switch (i) {
			case 0:
				input = Integer.parseInt(args[i]);
				break;
			case 1:
				hidden = Integer.parseInt(args[i]);
				break;
			case 2:
				output = Integer.parseInt(args[i]);
				break;
			case 3:
				patternFile = args[i];
				break;
			case 4:
				outputFile = args[i];
				break;
			case 5:
				testFile = args[i];
				break;
			case 6:
				learningRate = Double.parseDouble(args[i]);
				break;
			case 7:
				moment = Double.parseDouble(args[i]);
				break;
			case 8:
				cycles = Integer.parseInt(args[i]);
				break;
			case 9:
				numberOfPatterns = Integer.parseInt(args[i]);
				break;

			default:
				throw new Exception("Input argument exception");
			}
		}
		LabNet Net_1 = new LabNet(input, hidden, output, patternFile, outputFile);
		LabNet2 Net_2 = new LabNet2(input, hidden, output, patternFile, outputFile);
		Net_1.trainNetwork(patternFile, outputFile, learningRate, moment, cycles, numberOfPatterns);
		Net_2.trainNetwork(patternFile, "out_end.txt", learningRate, moment, cycles, numberOfPatterns);
		float[] input_parameters1 = new float[input];
		float[] input_parameters2 = new float[output];

		try (BufferedReader br = new BufferedReader(new FileReader(testFile))) {
			String line;
			BufferedWriter out = new BufferedWriter(new FileWriter("output_"+testFile));

			while ((line = br.readLine()) != null) {
				int j = 0;
				String[] myData = line.split(";");
				for (int i = 0; i < input + output; i++) {
					if (i < input) {
						input_parameters1[i] = Float.parseFloat(myData[i]);

					} else {
						input_parameters2[j++] = Float.parseFloat(myData[i]);
					}
				}

				float[] resultOutput = Net_1.getNetworkOutput(input_parameters1);
				float[] resultInput = Net_2.getNetworkOutput(input_parameters2);
				System.out.print("Rezultat je: ");
				for (int i = 0; i < resultInput.length; i++) {
					System.out.print(resultInput[i] + " ");
					out.write(resultInput[i] + ";");
				}
				for (int i = 0; i < resultOutput.length; i++) {
					System.out.print(resultOutput[i] + " ");
					out.write(resultOutput[i] + ";");
				}
				System.out.println();
				out.newLine();

			}
			out.close();
		}

	}

}
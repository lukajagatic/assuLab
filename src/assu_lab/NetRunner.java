package assu_lab;

public class NetRunner {

	public static void main(String[] args) throws Exception {
		int input = 0, hidden = 0, output = 0, cycles = 0, numberOfPatterns = 0;
		String patternFile = null, outputFile = null;
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
				learningRate = Double.parseDouble(args[i]);
				break;
			case 6:
				moment = Double.parseDouble(args[i]);
				break;
			case 7:
				cycles = Integer.parseInt(args[i]);
				break;
			case 8:
				numberOfPatterns = Integer.parseInt(args[i]);
				break;

			default:
				throw new Exception("Input argument exception");
			}
		}
		LabNet Net_1 = new LabNet(input, hidden, output, patternFile,
				outputFile);
		Net_1.trainFirstNetwork(patternFile, outputFile, learningRate, moment,
				cycles, numberOfPatterns);
		
		float[] input_parameters = new float[] { 1.0f, 1.0f };//TODO ADD MORE THAN ONE INPUT
		float[] result = Net_1.getFirstNetworkOutput(input_parameters);
		System.out.println("Rezultat 1 je: " + result[0]);

		Net_1.trainSecondNetwork(patternFile, "out_end", learningRate, moment,
				cycles, numberOfPatterns);
		
		float[] input_parameters2 = new float[] { 1.0f, 1.0f };
		float[] result2 = Net_1.getSecondNetworkOutput(input_parameters2);
		System.out.println("Rezultat 2 je: " + result2[0]);

	}

}
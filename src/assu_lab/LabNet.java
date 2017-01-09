package assu_lab;

import java.io.File;

import org.joone.engine.*;
import org.joone.engine.learning.*;
import org.joone.io.*;
import org.joone.net.NeuralNet;

public class LabNet implements NeuralNetListener {

	private LinearLayer input;
	private SigmoidLayer hidden;
	private SigmoidLayer output;
	private NeuralNet nnet;
	private Monitor monitor;
	private int outputNeurons;
	private int inputNeurons;

	public LabNet(int inputNeurons, int hiddenNeurons, int outputNeurons,
			String patternsFile, String outputFile) {
		this.input = new LinearLayer();
		this.hidden = new SigmoidLayer();
		this.output = new SigmoidLayer();
		this.input.setLayerName("input");
		this.hidden.setLayerName("hidden");
		this.output.setLayerName("output");
		this.outputNeurons = outputNeurons;
		this.inputNeurons = inputNeurons;
		/* sets their dimensions */
		this.input.setRows(inputNeurons);
		this.hidden.setRows(hiddenNeurons);
		this.output.setRows(outputNeurons);

		/*
		 * Now create the two Synapses
		 */
		FullSynapse synapse_IH = new FullSynapse(); /* input -> hidden conn. */
		FullSynapse synapse_HO = new FullSynapse(); /* hidden -> output conn. */

		synapse_IH.setName("IH");
		synapse_HO.setName("HO");
		/*
		 * Connect the input layer whit the hidden layer
		 */
		this.input.addOutputSynapse(synapse_IH);
		this.hidden.addInputSynapse(synapse_IH);
		/*
		 * Connect the hidden layer whit the output layer
		 */
		this.hidden.addOutputSynapse(synapse_HO);
		this.output.addInputSynapse(synapse_HO);

	}

	public void trainFirstNetwork(String patternsFile, String outputFile,
			double learningRate, double moment, int cycles, int numberOfPatterns) {
		TeachingSynapse trainer = new TeachingSynapse();

		FileInputSynapse inputStream = new FileInputSynapse();
		/* The first n columns contain the input values */
		String a = "";
		for (int i = 0; i < inputNeurons; i++) {
			a += i + 1;
			if ((i + 1) != inputNeurons)
				a += ",";
		}
		inputStream.setAdvancedColumnSelector(a);

		/* This is the file that contains the input data */
		inputStream.setInputFile(new File(patternsFile));
		this.input.addInputSynapse(inputStream);

		/*
		 * Setting of the file containing the desired responses, provided by a
		 * FileInputSynapse
		 */
		FileInputSynapse samples = new FileInputSynapse();
		samples.setInputFile(new File(patternsFile));
		/* The output values are on the third column of the file */
		String b = "";
		for (int i = inputNeurons; i < (inputNeurons + outputNeurons); i++) {
			b += i + 1;
			if (i != (inputNeurons + outputNeurons))
				b += ",";
		}
		samples.setAdvancedColumnSelector(b);

		trainer.setDesired(samples);

		/* Creates the error output file */
		FileOutputSynapse error = new FileOutputSynapse();
		error.setFileName(outputFile);
		// error.setBuffered(false);
		trainer.addResultSynapse(error);

		/* Connects the Teacher to the last layer of the net */
		this.output.addOutputSynapse(trainer);

		this.nnet = new NeuralNet();
		this.nnet.addLayer(this.input, NeuralNet.INPUT_LAYER);
		this.nnet.addLayer(this.hidden, NeuralNet.HIDDEN_LAYER);
		this.nnet.addLayer(this.output, NeuralNet.OUTPUT_LAYER);

		this.nnet.setTeacher(trainer);
		// Gets the Monitor object and set the learning parameters
		this.monitor = nnet.getMonitor();
		this.monitor.setLearningRate(learningRate);
		this.monitor.setMomentum(moment);

		/*
		 * The application registers itself as monitor's listener so it can
		 * receive the notifications of termination from the net.
		 */
		this.monitor.addNeuralNetListener(this);

		this.monitor.setTrainingPatterns(numberOfPatterns); /*
															 * # of rows
															 * (patterns)
															 * contained in the
															 * input file
															 */
		this.monitor.setTotCicles(cycles); /*
											 * How many times the net must be
											 * trained on the input patterns
											 */
		this.monitor.setLearning(true); /* The net must be trained */
		this.nnet.getMonitor().setSingleThreadMode(true); // wait until learning
															// is finished
		this.nnet.go(true); /*
							 * The net starts the training job, true -
							 * synchronized
							 */
	}

	public void trainSecondNetwork(String patternsFile, String outputFile,
			double learningRate, double moment, int cycles, int numberOfPatterns) {
		TeachingSynapse trainer = new TeachingSynapse();

		FileInputSynapse inputStream = new FileInputSynapse();
		/* The first n columns contain the input values */
		String a = "";
		for (int i = outputNeurons; i < (outputNeurons + inputNeurons); i++) {
			a += i + 1;
			if ((i + 1) != (outputNeurons + inputNeurons))
				a += ",";
		}
		inputStream.setAdvancedColumnSelector(a);

		/* This is the file that contains the input data */
		inputStream.setInputFile(new File(patternsFile));
		this.input.addInputSynapse(inputStream);

		/*
		 * Setting of the file containing the desired responses, provided by a
		 * FileInputSynapse
		 */
		FileInputSynapse samples = new FileInputSynapse();
		samples.setInputFile(new File(patternsFile));
		/* The output values are on the third column of the file */
		String b = "";
		for (int i = 0; i < outputNeurons; i++) {
			b += i + 1;
			if (i != outputNeurons)
				b += ",";
		}
		samples.setAdvancedColumnSelector(b);

		trainer.setDesired(samples);

		/* Creates the error output file */
		FileOutputSynapse error = new FileOutputSynapse();
		error.setFileName(outputFile);
		// error.setBuffered(false);
		trainer.addResultSynapse(error);

		/* Connects the Teacher to the last layer of the net */
		this.output.addOutputSynapse(trainer);

		this.nnet = new NeuralNet();
		this.nnet.addLayer(this.input, NeuralNet.INPUT_LAYER);
		this.nnet.addLayer(this.hidden, NeuralNet.HIDDEN_LAYER);
		this.nnet.addLayer(this.output, NeuralNet.OUTPUT_LAYER);

		this.nnet.setTeacher(trainer);
		// Gets the Monitor object and set the learning parameters
		this.monitor = nnet.getMonitor();
		this.monitor.setLearningRate(learningRate);
		this.monitor.setMomentum(moment);

		/*
		 * The application registers itself as monitor's listener so it can
		 * receive the notifications of termination from the net.
		 */
		this.monitor.addNeuralNetListener(this);

		this.monitor.setTrainingPatterns(numberOfPatterns); /*
															 * # of rows
															 * (patterns)
															 * contained in the
															 * input file
															 */
		this.monitor.setTotCicles(cycles); /*
											 * How many times the net must be
											 * trained on the input patterns
											 */
		this.monitor.setLearning(true); /* The net must be trained */
		this.nnet.getMonitor().setSingleThreadMode(true); // wait until learning
															// is finished
		this.nnet.go(true); /*
							 * The net starts the training job, true -
							 * synchronized
							 */
	}

	public float[] getFirstNetworkOutput(float[] input_parameters) {
		Layer input = this.nnet.getInputLayer();
		input.removeAllInputs();
		MemoryInputSynapse memInp = new MemoryInputSynapse();
		memInp.setFirstRow(1);
		memInp.setAdvancedColumnSelector("1-" + inputNeurons);
		input.addInputSynapse(memInp);
		double[][] codedPattern = new double[1][inputNeurons];
		for (int i = 0; i < input_parameters.length; i++) {
			codedPattern[0][i] = input_parameters[i];
		}
		memInp.setInputArray(codedPattern);
		/*
		 * We get the last layer of the net (the output layer), then remove all
		 * the output synapses attached to it and attach a MemoryOutputSynapse
		 */
		Layer output = this.nnet.getOutputLayer();
		// Remove all the output synapses attached to it...
		output.removeAllOutputs();
		// ...and attach a MemoryOutputSynapse
		MemoryOutputSynapse memOut = new MemoryOutputSynapse();
		output.addOutputSynapse(memOut);
		// Now we interrogate the net
		this.nnet.getMonitor().setTotCicles(1);
		this.nnet.getMonitor().setTrainingPatterns(1);
		this.nnet.getMonitor().setLearning(false);
		this.nnet.getMonitor().setSingleThreadMode(true);
		this.nnet.go(true);// true - synchronized

		float[] resultPattern = new float[outputNeurons];

		for (int i = 0; i < 1; i++) {
			// Read the next pattern and print out it
			double[] pattern = memOut.getNextPattern();
			for (int j = 0; j < pattern.length; j++) {
				resultPattern[j] = (float) pattern[j];
			}
		}
		nnet.stop();
		return resultPattern;
	}

	public float[] getSecondNetworkOutput(float[] input_parameters) {
		Layer input = this.nnet.getInputLayer();
		input.removeAllInputs();
		MemoryInputSynapse memInp = new MemoryInputSynapse();
		memInp.setFirstRow(1);
		memInp.setAdvancedColumnSelector("1-" + inputNeurons);
		input.addInputSynapse(memInp);
		double[][] codedPattern = new double[1][inputNeurons];
		for (int i = 0; i < input_parameters.length; i++) {
			codedPattern[0][i] = input_parameters[i];
		}
		memInp.setInputArray(codedPattern);
		/*
		 * We get the last layer of the net (the output layer), then remove all
		 * the output synapses attached to it and attach a MemoryOutputSynapse
		 */
		Layer output = this.nnet.getOutputLayer();
		// Remove all the output synapses attached to it...
		output.removeAllOutputs();
		// ...and attach a MemoryOutputSynapse
		MemoryOutputSynapse memOut = new MemoryOutputSynapse();
		output.addOutputSynapse(memOut);
		// Now we interrogate the net
		this.nnet.getMonitor().setTotCicles(1);
		this.nnet.getMonitor().setTrainingPatterns(1);
		this.nnet.getMonitor().setLearning(false);
		this.nnet.getMonitor().setSingleThreadMode(true);
		this.nnet.go(true);// true - synchronized

		float[] resultPattern = new float[outputNeurons];

		for (int i = 0; i < 1; i++) {
			// Read the next pattern and print out it
			double[] pattern = memOut.getNextPattern();
			for (int j = 0; j < pattern.length; j++) {
				resultPattern[j] = (float) pattern[j];
			}
		}
		nnet.stop();
		return resultPattern;
	}

	
	public float[] trainAndTestFirst(String patternsFile, String outputFile,
			double learningRate, double moment, int cycles,
			int numberOfPatterns, float[] input_parameters) {
		this.trainFirstNetwork(patternsFile, outputFile, learningRate, moment,
				cycles, numberOfPatterns);
		return this.getFirstNetworkOutput(input_parameters);
	}
	
	public float[] trainAndTestSecond(String patternsFile, String outputFile,
			double learningRate, double moment, int cycles,
			int numberOfPatterns, float[] input_parameters) {
		this.trainSecondNetwork(patternsFile, outputFile, learningRate, moment,
				cycles, numberOfPatterns);
		return this.getSecondNetworkOutput(input_parameters);
	}

	public void netStopped(NeuralNetEvent e) {
		System.out.println("Network finished");
	}

	public void cicleTerminated(NeuralNetEvent e) {
	}

	public void netStarted(NeuralNetEvent e) {
		System.out.println("Network started");
	}

	public void errorChanged(NeuralNetEvent e) {
		Monitor mon = (Monitor) e.getSource();
		/* We want print the results every 200 cycles */
		if (mon.getCurrentCicle() % 200 == 0)
			System.out.println(mon.getCurrentCicle()
					+ " epochs remaining - RMSE = " + mon.getGlobalError());
	}

	public void netStoppedError(NeuralNetEvent e, String error) {
	}

}

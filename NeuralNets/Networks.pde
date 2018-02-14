
abstract class Network {
  protected boolean initialized = false;
  protected boolean training = false;
  protected boolean momentum = true;

  protected int iteration;
  protected int numInputs, numOutputs, numLayers;
  protected int currentIteration;
  protected int nodesPerLayer[];
  public int x, y;

  public double currentCost, minCost;
  public double learningRate = .15;
  public double alpha = 0.9;

  protected ArrayList<Matrix> layerWeights;
  protected ArrayList<Matrix> layerBiases;
  protected ArrayList<Matrix> lastdCdW, lastdCdB;
  protected double[] lastInputs;
  protected double[] lastExpected;

  int seed = 8675309;

  public void Initialize() {
    randomSeed(seed);
    layerWeights = new ArrayList<Matrix>();
    layerBiases = new ArrayList<Matrix>();
    for (int i = 0; i <= numLayers; i++) {
      int rows, cols;
      if (i == 0) { 
        rows = numInputs;
      } else { 
        rows = nodesPerLayer[i - 1];
      }
      if (i == numLayers) { 
        cols = numOutputs;
      } else {
        cols = nodesPerLayer[i];
      }

      Matrix m = new Matrix(rows, cols);
      Matrix n = new Matrix(1, cols);
      for (int j = 0; j < rows; j++) {
        for (int k = 0; k < cols; k++) {
          m.set(j, k, random(-1, 1));
          n.set(0, k, random(-1, 1));
        }
      }
      layerWeights.add(m);
      layerBiases.add(n);
    }
    iteration = 0;
    initialized = true;
  }

  public void RandomizeWeights() {
    //Use the matrix in layer i
    for (int i = 0; i < layerWeights.size(); i++) {
      //Iterate through the rows
      for (int j = 0; j < layerWeights.get(i).getRowDimension(); j++) {
        //Iterate through the columns
        for (int k = 0; k < layerWeights.get(i).getColumnDimension(); k++) {
          //Use a random double
          layerWeights.get(i).set(j, k, random(-1, 1));
        }
      }
    }
  }
  public void RandomizeBiases() {
    //Use the matrix in layer i
    for (int i = 0; i < layerBiases.size(); i++) {
      //Iterate through the rows
      for (int j = 0; j < layerBiases.get(i).getRowDimension(); j++) {
        layerWeights.get(i).set(0, j, random(-1, 1));
      }
    }
  }
  public void SetInputSize(int n) {
    numInputs = n;
  }
  public void SetOutputSize(int n) {
    numOutputs = n;
  }
  public void SetNumLayers(int n) {
    numLayers = n;
  }
  public void SetNodesInEachLayer(int[] n) {
    nodesPerLayer = n;
  }
  public void SetLearningRate(double d) {
    learningRate = d;
  }
  public void SetHiddenLayerWeights(ArrayList<Matrix> w) {
    layerWeights = w;
  }
  public int GetNumInputNodes() {
    return numInputs;
  }
  public int GetNumLayers() {
    return numLayers;
  }
  public int[] GetNodesPerLayer() {
    return nodesPerLayer;
  }
  public int GetNumOutputNodes() {
    return numOutputs;
  }
  public ArrayList<Matrix> GetWeights() {
    return layerWeights;
  }
  public boolean IsTraining() { 
    return training;
  }
  public boolean IsInitialized() { 
    return initialized;
  }
  public double GetCurrentCost() { 
    return currentCost;
  }
  public int GetCurrentIteration() { 
    return currentIteration;
  }
  public int maxNodesInALayer() {
    int max = 0;
    for (int i = 0; nodesPerLayer != null && i < nodesPerLayer.length; i++) {
      if (nodesPerLayer[i]> max) {
        max = nodesPerLayer[i];
      }
    }
    if (numInputs > max) {
      max = numInputs;
    }
    if (numOutputs > max) {
      max = numOutputs;
    }
    return max;
  }

  public void draw() {
    if (initialized) {
      int currentLayer = 0;
      int h = 400;
      int w = 500;
      // Draw input values
      for (int i = 0; lastInputs != null && i < lastInputs.length; i++) {
        textSize(10);
        text(lastInputs[i] + "", x + currentLayer * (w / (2 + numLayers)) - 30, 
          y + (i + 1) * (h / (numInputs  + 2 + 1)) + 5);
      }

      // Draw connections and weights.      
      for (int layer = 0; layer < layerWeights.size(); layer++) {
        for (int i = 0; i < layerWeights.get(layer).getRowDimension(); i++) {
          for (int j = 0; j < layerWeights.get(layer).getColumnDimension(); j++) {
            double weight = layerWeights.get(layer).get(i, j);
            int c = color(Math.abs((int)(weight * 4127 ) % 255), 
              Math.abs((int)(weight * 13291) % 255), 
              Math.abs((int)(weight * 30319) % 255));
            fill(c);
            stroke(c);
            line(x + layer * (w / (2 + numLayers)), 
              y + (i + 1) * (h / (layerWeights.get(layer).getRowDimension() + 2)), 
              x + (layer + 1) * (w / (2 + numLayers)), 
              y + (j + 1) * (h / (layerWeights.get(layer).getColumnDimension() + 2)));
            textSize(10);
            fill(c);
            text((layerWeights.get(layer).get(i, j) + "").substring(0, 5), 
              x + (layer * (w / (2 + numLayers)) + (layer + 1) * (w / (2 + numLayers))) / 2, 
              y + ((i + 1) * (h / (layerWeights.get(layer).getRowDimension() + 2)) + 
              (j + 1) * (h / (layerWeights.get(layer).getColumnDimension() + 2))) / 2);
          }
        }
      }
      for (int i = 0; i < numInputs; i++) {
        fill(0, 0, 255);
        stroke(0);
        ellipse(x + currentLayer * (w / (2 + numLayers)), 
          y + (i + 1) * (h / (numInputs + 2)), 20, 20);
      }
      currentLayer++;
      // Draw layer nodes
      for (int layer = 0; layer < numLayers; layer++) {
        for (int i = 0; i < nodesPerLayer[layer]; i++) {
          fill(0, 0, 255);
          stroke(0);
          text("b="+(layerBiases.get(layer).get(0, i)+"").substring(0, 5), x + currentLayer * (w / (2 + numLayers)) - 10, 
            y + (i + 1) * (h / (nodesPerLayer[layer] + 2)) - 10);

          ellipse(x + currentLayer * (w / (2 + numLayers)), 
            y + (i + 1) * (h / (nodesPerLayer[layer] + 2)), 20, 20);
        }
        currentLayer++;
      }
      // Draw outputs
      for (int i = 0; i < numOutputs; i++) {
        fill(0, 0, 255);
        stroke(0);
        text("b="+(layerBiases.get(currentLayer-1).get(0, i)+"").substring(0, 5), 
          x + currentLayer * (w / (2 + numLayers) - 10), 
          y + (i + 1) * (h / (numOutputs + 2)) - 10);

        ellipse(x + currentLayer * (w / (2 + numLayers)), 
          y + (i + 1) * (h / (numOutputs + 2)), 20, 20);
      }
      for (int i = 0; lastExpected != null && i < lastExpected.length; i++) {
        fill(0);
        text(lastExpected[i] + "", x + currentLayer * (w / (2 + numLayers)) + 15, 
          y + (i + 1) * (h / (numOutputs + 2)) + 5);
      }
      currentLayer++;
    }
  }
}

class SigmoidNetwork extends Network {
  public SigmoidNetwork() {
  }
  private ArrayList<Matrix> populateDeltas(Matrix actual, Matrix predicted, 
    ArrayList<Matrix> valuesBeforeActivation) {
    ArrayList<Matrix> deltas = new ArrayList<Matrix>();

    for (int i = numLayers; i >= 0; i--) {
      Matrix delta;
      if (i == numLayers) {
        delta = predicted.copy();
        delta = delta.minus(actual);
      } else {
        delta = deltas.get(0).copy();
        Matrix w = layerWeights.get(i + 1).copy();
        w = w.transpose();
        delta = delta.times(w);
      }
      Matrix sigmoidPrimes = SigmoidNeuron.ActivationPrime(valuesBeforeActivation.get(i).copy());
      delta = ElementwiseMultiplication(delta, sigmoidPrimes);
      deltas.add(0, delta.copy());
    }    

    return deltas;
  }

  public void Train(double[][] inputs, double[][] expected) {
    lastInputs = inputs[0];
    lastExpected = expected[0];
    minCost = cost(inputs, expected);
    Matrix inputsMatrix = Matrix.constructWithCopy(inputs);
    Matrix correctResultsMatrix = Matrix.constructWithCopy(expected);      

    ArrayList<Matrix> valuesBeforeActivation = new ArrayList<Matrix>();
    ArrayList<Matrix> activationValues = new ArrayList<Matrix>();

    Matrix estimatedResultsMatrix = ForwardAllTraining(inputsMatrix, 
      valuesBeforeActivation, activationValues);      

    ArrayList<Matrix> deltas = populateDeltas(correctResultsMatrix, estimatedResultsMatrix, 
      valuesBeforeActivation);

    ArrayList<Matrix> dCost_dWeights = costFunctionPrime(deltas, activationValues);
    lastdCdW = new ArrayList<Matrix>();
    lastdCdB = new ArrayList<Matrix>();

    for (int i = 0; i < layerWeights.size(); i++) {
      Matrix dCost_dWeight = dCost_dWeights.get(i).copy();
      Matrix dCost_dBias = deltas.get(i).copy();

      dCost_dWeight = dCost_dWeight.times(learningRate);
      dCost_dBias = dCost_dBias.times(learningRate);

      lastdCdW.add(dCost_dWeight.copy());
      lastdCdB.add(dCost_dBias.copy());

      if (iteration != 0 && momentum) {
        dCost_dWeight = dCost_dWeight.plus(lastdCdW.get(i).times(alpha)); 
        dCost_dBias = dCost_dBias.plus(lastdCdB.get(i).times(alpha));
      }

      layerWeights.set(i, 
        layerWeights.get(i).minus(dCost_dWeight));
      layerBiases.set(i, 
        layerBiases.get(i).minus(dCost_dBias));
    }

    currentCost = cost(inputs, expected);
    iteration++;
  }
  public double[] Forward(double[] inputs) {
    double[] results = new double[numOutputs];

    Matrix inputMatrix = new Matrix(1, inputs.length);  
    for (int i = 0; i < inputs.length; i++) {
      inputMatrix.set(0, i, inputs[i]);
    }
    Matrix resultsMatrix = inputMatrix.copy();
    for (int i = 0; i <= numLayers; i++) {
      //Get the weights for the current layer
      Matrix weights = layerWeights.get(i);
      Matrix biases = layerBiases.get(i);
      //Right multiply the results by the weights
      resultsMatrix = resultsMatrix.times(weights);
      resultsMatrix = resultsMatrix.plus(biases);
      resultsMatrix = SigmoidNeuron.Activate(resultsMatrix);
    }    
    //The results will be in a row matrix; get the double array copy of that and return;
    results = resultsMatrix.getRowPackedCopy();
    return results;
  }
  /*
   * Solves all inputs in the neural network, used for training.
   */
  private Matrix ForwardAllTraining(Matrix inputs, ArrayList<Matrix> valuesBeforeActivation, 
    ArrayList<Matrix> activationValues) {
    Matrix results = inputs.copy();
    activationValues.add(results.copy());
    for (int i = 0; i <= numLayers; i++) {
      //Get the weights for the current layer
      Matrix weights = layerWeights.get(i);
      Matrix biases = layerBiases.get(i);
      //Right multiply the results by the weights
      results = results.times(weights.copy());
      results = results.plus(biases.copy());
      valuesBeforeActivation.add(results.copy());
      //Activate each neuron
      results = SigmoidNeuron.Activate(results);
      activationValues.add(results.copy());
    }  
    //Each row will represent the outputs for the corresponding inputs row
    activationValues.remove(activationValues.size() - 1);
    return results;
  }
  public double cost(double[] input, double output[]) {
    double[] actual = Forward(input);
    double c = 0;
    for (int i = 0; i < actual.length; i++) {
      c += Math.pow(output[i] - actual[i], 2);
    }
    return c / 2;
  }
  public double error(double[] input, double output[]) {
    double[] actual = Forward(input);
    double c = 0;
    for (int i = 0; i < actual.length; i++) {
      c += abs((float)(output[i] - actual[i]));
    }
    return c;
  }
  public double cost(double[][] input, double output[][]) {
    double c = 0;
    for (int i = 0; i < input.length; i++) {
      double[] results = Forward(input[i]);      
      for (int j = 0; j < results.length; j++) {
        c += Math.pow(output[i][j] - results[j], 2);
      }
    }
    c /= 2.0;

    return c;
  }
  /*
   * Used in backpropagation to find the gradient
   */
  private ArrayList<Matrix> costFunctionPrime(ArrayList<Matrix> deltas, ArrayList<Matrix> activationValues) {
    ArrayList<Matrix> dCost_dWeights = new ArrayList<Matrix>();
    for (int i = 0; i < layerWeights.size(); i++) {  
      Matrix dCost_dWeight = activationValues.get(i).copy().transpose();          
      dCost_dWeight = dCost_dWeight.times(deltas.get(i).copy());          
      dCost_dWeights.add(dCost_dWeight);
      //printMatrix(dCost_dWeight);
    }    
    return dCost_dWeights;
  }
}

class MPNetwork extends Network { 
  public MPNetwork() {
  }
  public void Train(double[] inputs, double[] expected) {
    double[] estimates = Forward(inputs);
    for (int i = 0; i < estimates.length; i++) {
      expected[i] = expected[i] - estimates[i];
    }
    for (int i = 0; i <= numLayers; i++) {
      int rows, cols;
      if (i == 0) rows = numInputs;
      else rows = nodesPerLayer[i - 1];
      if (i == numLayers) cols = numOutputs;
      else cols = nodesPerLayer[i];
      //leftLayer
      for (int j = 0; j < rows; j++) {
        //rightLayer
        for (int k = 0; k < cols; k++) {
          layerWeights.get(i).set(j, k, 
            layerWeights.get(i).get(j, k) + learningRate * expected[k] * inputs[j]);
        }
      }
    }
  }
  public double[] Forward(double[] inputs) {
    double[] results = new double[numOutputs];

    Matrix inputMatrix = new Matrix(1, inputs.length);  
    for (int i = 0; i < inputs.length; i++) {
      inputMatrix.set(0, i, inputs[i]);
    }

    Matrix resultsMatrix = inputMatrix.copy();
    for (int i = 0; i <= numLayers; i++) {
      //Get the weights for the current layer
      Matrix weights = layerWeights.get(i);
      //Right multiply the results by the weights
      resultsMatrix = resultsMatrix.times(weights);
      resultsMatrix = MPneuron.Activate(resultsMatrix);
    }    
    //The results will be in a row matrix; get the double array copy of that and return;
    results = resultsMatrix.getRowPackedCopy();
    return results;
  }
  public double cost(double[] input, double output[]) {
    double[] actual = Forward(input);
    double c = 0;
    for (int i = 0; i < actual.length; i++) {
      c += Math.pow(output[i] - actual[i], 2);
    }
    return c;
  }
  public double cost(double[][] input, double output[][]) {
    double c = 0;
    for (int i = 0; i < input.length; i++) {
      double[] results = Forward(input[i]);      
      for (int j = 0; j < results.length; j++) {
        c += Math.pow(output[i][j] - results[j], 2);
      }
    }
    c /= 2.0;

    return c;
  }
}
import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import Jama.*; 
import Jama.examples.*; 
import Jama.test.*; 
import Jama.util.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class NeuralNets extends PApplet {







int screenWidth = 900;
int screenHeight = 500;
int iteration = 0;
double biggestErrorForEpoch = 0;

SigmoidNetwork nn;
TextInput numInputs, numOutputs, numLayers, nodesPerLayer;
IntSlider speed;
ArrayList<TextInput> inputs = new ArrayList<TextInput>();
Rectangle init;
Rectangle begin;

public void start() {

  numInputs = new TextInput(5, 50, 50, 25);
  numOutputs = new TextInput(5, 105, 50, 25);
  numLayers = new TextInput(5, 155, 50, 25);
  nodesPerLayer = new TextInput(5, 205, 50, 25);
  init = new Rectangle(5, 280, 130, 30);
  init.c = color(255);
  begin = new Rectangle(150, 280, 130, 30);
  begin.c = color(255);
  speed = new IntSlider("Speed", 0, 2000, 20, 430, 200);
  nn = new SigmoidNetwork();
  nn.x = 340;
  nn.y = 5;
}
public void setup() {
  
  
  background(255);
}

public void draw() {
  background(150);
  stroke(255);


  fill(0);
  textSize(20);
  text("Number of inputs", 5, 25 + 20);
  numInputs.draw();
  fill(0);
  textSize(20);
  text("Number of outputs", 5, 80 + 20);
  numOutputs.draw();
  fill(0);
  textSize(20);
  text("Number of hidden layers", 5, 130 + 20);
  numLayers.draw();
  fill(0);
  textSize(20);
  text("Nodes per hidden layer", 5, 180 + 20);
  nodesPerLayer.draw();

  init.draw();
  fill(0);
  textSize(22);
  text("Initialize", 20, 305);
  if (nn.initialized) {
    begin.draw();
    fill(0);
    textSize(22);
    if (nn.training) text("Stop", 190, 305);
    else text("Begin", 190, 305);
  }
  textSize(16);
  text("Current cost: " + nn.currentCost, 20, 350);
  text("Largest cost this epoch: " + biggestErrorForEpoch, 20, 380);
  text("Epoch " + iteration / 16 + ", iteration " + iteration, 20, 410);
  speed.draw();
  nn.draw();
}
public void mousePressed() {
  numInputs.mousePressed();
  numOutputs.mousePressed();
  numLayers.mousePressed();
  nodesPerLayer.mousePressed();
  speed.mousePressed();
  for (int i = 0; i < inputs.size(); i++) {
    inputs.get(i).mousePressed();
  }
  if (init.contains(mouseX, mouseY) && !nn.training) {
    iteration = 0;
    nn.SetInputSize(Integer.parseInt(numInputs.text));
    nn.SetOutputSize(Integer.parseInt(numOutputs.text));
    nn.SetNumLayers(Integer.parseInt(numLayers.text));
    int[] layerSizes = new int[Integer.parseInt(numLayers.text)];
    int temp = 0;
    int spot = 0;
    if (Integer.parseInt(numLayers.text) > 0) {
      for (int i = 0; i < nodesPerLayer.text.length(); i++) {
        if (isInteger(nodesPerLayer.text.substring(i, i+1))) {
          temp *= 10;
          temp += Integer.parseInt(nodesPerLayer.text.substring(i, i+1));
        } else {
          layerSizes[spot] = temp;
          spot++;
          temp = 0;
        }
      }
      layerSizes[spot] = temp;
      nn.SetNodesInEachLayer(layerSizes);
    }
    nn.Initialize();
  }
  if (begin.contains(mouseX, mouseY)) {
    nn.training = !nn.training;
    if (nn.training)
      thread("threadedTraining");
  }
}
public void mouseDragged() {
  speed.mouseDragged();
}
public void mouseReleased() {
  speed.mouseReleased();
}
public void keyPressed() {
  numInputs.keyPressed();
  numOutputs.keyPressed();
  numLayers.keyPressed();
  nodesPerLayer.keyPressed();
  for (int i = 0; i < inputs.size(); i++) {
    inputs.get(i).keyPressed();
  }
}

public void threadedTraining() {
  while (nn.training) {
    if (iteration % 16 == 0) { 
      biggestErrorForEpoch = 0;
    }
    double[] input = project1InputSeeded(iteration);
    double[] output = project1Output(input);
    if (nn.IsInitialized()) {
      nn.Train(new double[][]{input}, 
        new double[][] {output});
    }
    if (nn.currentCost > biggestErrorForEpoch){
      biggestErrorForEpoch = nn.currentCost;
    }
    
    if (iteration % 16 == 15) { 
      if (biggestErrorForEpoch <= 0.05f) {
        nn.training = false;
        return;
      }
    }
    
    iteration++;
    delay(speed.getValue());
  }
}

public double[] randomInputHomework1Question1() {
  //bias, x, y, z
  return new double[] {1, (int)pow(-1, (int)random(2)), (int)pow(-1, (int)random(2)), (int)pow(-1, (int)random(2))};
}
public double[] outputHomework1Question1(double[] in) {
  if (in[1] == -1 && in[2] == 1) {
    return new double[]{in[3]};
  } else return new double[]{-1};
}
public double[] randomInputHomework1Question2() {
  //bias, u, v, w, x where uv and wx are binary numbers
  return new double[] {1, (int)random(2), (int)random(2), (int)random(2), (int)random(2)};
}
public double[] outputHomework1Question2(double[] in) {
  //yz (as binary number)
  return new double[] {(in[2] == 1 ^ in[4] == 1) ^ in[1] == 1 ^ in [3] == 1? 1 : 0, 
    in[4] == 1 ^ in[2] == 1  ? 1 : 0  };
}
public double[] project1Input() {
  // b, x1, x2, x3, x4, x5, x6, x7, x8
  return new double[] {(int)random(2), 
    (int)random(2), (int)random(2), 
    (int)random(2)};
}
public double[] project1InputSeeded(int i) {
  // b, x1, x2, x3, x4, x5, x6, x7, x8
  int q = (i % 16) / 8;
  int w = (i % 8) / 4;
  int r = (i % 4) / 2;
  return new double[] {q, w, r, i % 2};
}
public double[] project1Output(double[] input) {
  int num1s = 0;
  for (int i = 0; i < input.length; i++) {
    if (input[i] == 1) num1s++;
  }
  return new double[] {num1s % 2 == 1 ? 1 : 0};
}

abstract class Network {
  protected boolean initialized = false;
  protected boolean training = false;

  protected int iteration;
  protected int numInputs, numOutputs, numLayers;
  protected int currentIteration;
  protected int nodesPerLayer[];
  public int x, y;

  public double currentCost, minCost;
  public double learningRate = .35f;
  public double alpha = 0.9f;

  protected ArrayList<Matrix> layerWeights;
  protected ArrayList<Matrix> layerBiases;
  protected ArrayList<Matrix> lastdCdW, lastdCdB;
  protected double[] lastInputs;
  protected double[] lastExpected;


  public void Initialize() {
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
          text("b="+(layerBiases.get(layer).get(0, i)+"").substring(0, 5),x + currentLayer * (w / (2 + numLayers)) - 10, 
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
      
      if(iteration != 0){
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
  public double cost(double[][] input, double output[][]) {
    double c = 0;
    for (int i = 0; i < input.length; i++) {
      double[] results = Forward(input[i]);      
      for (int j = 0; j < results.length; j++) {
        c += Math.pow(output[i][j] - results[j], 2);
      }
    }
    c /= 2.0f;

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
    c /= 2.0f;

    return c;
  }
}

static class Perceptron{
  public static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       if(m.get(i, j) >= 0) m.set(i, j, 1);
       else m.set(i, j, 0);
     }
    }
    return m;
  }
}

static class MPneuron{  
  public static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       if(m.get(i, j) >= 0) m.set(i, j, 1);
       else m.set(i, j, -1);
     }
    }
    return m;
  }  
}

static class SigmoidNeuron{  
  public static Matrix Activate(Matrix m){
    for(int i = 0; i < m.getRowDimension(); i++){
     for(int j = 0; j < m.getColumnDimension(); j++){
       m.set(i, j, 1.0f / (1 + Math.exp(-m.get(i, j))));
     }
    }
    return m;
  }  
  public static Matrix ActivationPrime(Matrix m){
    m = Activate(m);
    for(int i = 0; i < m.getRowDimension(); i++){
      for(int j = 0; j < m.getColumnDimension(); j++){
        m.set(i, j, 
          m.get(i, j) * (1 - m.get(i, j)));
      }
    }
    /*
    for(int i = 0; i < m.getRowDimension(); i++){
      for(int j = 0; j < m.getColumnDimension(); j++){
        m.set(i, j, 
        (Math.exp(-m.get(i, j))) / (Math.pow(1 + Math.exp(-m.get(i, j)), 2))
        );
      }
    }
    */
    return m;
  }
}


class TextInput{
 public boolean selected = false;
 public String text = "0"; 
 public Rectangle r;
 public TextInput(int x, int y, int width, int height){
   r = new Rectangle(x, y, width, height);
   r.c = color(255);
 }
 public void draw(){
   noStroke();
   r.draw(); 
   fill(0);
   textSize(r.height - 5);
   text(text, r.x + 1, r.y + r.height);
   if(selected){
     noFill();
     stroke(0, 255, 0);
     rect(r.x, r.y, r.width, r.height);
   }
   else{
    noFill();
    stroke(0);
    rect(r.x, r.y, r.width, r.height);
   }
 }
 public void mousePressed(){
  if(r.contains(mouseX, mouseY)){
    selected = true;
  }
  else selected = false;
 }
 public void keyPressed() {
   if(selected){
    if (keyCode == BACKSPACE) {
      if (text.length() > 0) {
        text = text.substring(0, text.length()-1);
      }
    } else if (keyCode == DELETE) {
      text = "";
    } else if (keyCode != SHIFT && keyCode != CONTROL && keyCode != ALT
      && textWidth(text + key) < r.width) {
      text = text + key;
    }
   }
}
}

class IntSlider {
  int min, max, x, y, size;
  String name;
  Rectangle background;
  Rectangle slider; 
  boolean holding = false;
  int sliderSize;
  public IntSlider(String name, int min, int max, int x, int y, int size) {
    this.min = min;
    this.max = max;
    this.name = name;
    this.x = x;
    this.y = y; 
    this.size = size;
    sliderSize = size / 15;
    background = new Rectangle(x, y, size, sliderSize);
    slider = new Rectangle(x, y - 5, sliderSize / 2, sliderSize + 10);
    slider.c = color(0);
    background.c = color(0);
  }
  public void setMin(int n){
    min = n;
  }
  public void setMax(int n){
    max = n;
  }
  public int getValue() {
    int closest = 0;
    float dist = distance(x, y, slider.x, slider.y);
    for (int i = 1; i <= max - min; i += 1) {
      float d = distance(slider.x, slider.y, x + i * (size / (float)(max - min)), y);
      if (dist > d) {
        dist =  d;
        closest = i;
      }
    }
    return min + closest;
  }
  public void draw() {

    background.draw();
    slider.draw();
    fill(255);
    textSize(size / 20);
    text(min, x - sliderSize * (min + "").length() - 5, y + sliderSize);
    text(max, x + size + 5, y + sliderSize);
    text(getValue(), slider.x, slider.y + slider.height + sliderSize);
  }

  public void mousePressed() {
    if (slider.contains(mouseX, mouseY)) {
      holding = true;
    }
  }
  public void mouseDragged() {
    if (holding) {
      slider.x = mouseX;
      if (slider.x < x) slider.x = x;
      if (slider.x > x + size) slider.x = x + size;
    }
  }
  public void mouseReleased() {
    holding = false;
  }
}

class Point{
  public int x, y;
  public Circle circle;
  public int c;
  public Point(int x, int y){
   this.x = x; 
   this.y = y;
   circle = new Circle(x, y, 3, 3);
   circle.c = color(0);
  }
  public void draw(){
   circle.draw();
  }
}
class MoveablePoint extends Point{
 public Circle circle;
 public boolean holding = false;
 public Rectangle bounds;
 public MoveablePoint(int x, int y, Rectangle bounds){
  super(x, y);
  this.x = x;
  this.y = y;
  circle = new Circle(x, y, 12, 12);
  circle.c = color(0, 0, 255);
  this.bounds = bounds;
 }
 public Point toPoint(){
  return new Point(x, y);
 }
 public void draw(){
   circle.c = color(0, 0, 255);
   if(holding) circle.c = color(255, 0, 0);
   circle.draw();
 }
 public void mousePressed(){
   if(circle.contains(mouseX, mouseY)){
     holding = true;
   }
 }
 public void mouseDragged(){
   if(holding){
     x = mouseX;
     y = mouseY;
     circle.x = mouseX;
     circle.y = mouseY;
     if(circle.x < bounds.x){
       circle.x = bounds.x;
       x = bounds.x;
     }
     if(circle.y < bounds.y){
       circle.y = bounds.y;
       y = bounds.y;  
     }
     if(circle.x > bounds.x + bounds.width){
       circle.x = bounds.x + bounds.width;
       x = bounds.x + bounds.width;
     }
     if(circle.y > bounds.y + bounds.height){
       circle.y = bounds.y + bounds.height;
       y = bounds.y + bounds.height;
     }
   }
 }
 public void mouseReleased(){
   holding = false;
 }
}

class Rectangle{
  public int x, y, width, height;
  public int c;
  public Rectangle(int x, int y, int width, int height){
   this.x = x;
   this.y = y;
   this.width = width;
   this.height = height;
  }
  public void draw(){
    fill(c);
    rect(x, y, width, height);
    noFill();
    stroke(0);
    rect(this.x, this.y, this.width, this.height);
  }
  public boolean contains(float x, float y){
    return x >= this.x && x <= this.x + width && y >= this.y && y <= this.y + height;
  }
}

class Circle{
  public int x, y, width, height;
  public int c;
  public Circle(int x, int y, int width, int height){
   this.x = x;
   this.y = y;
   this.width = width;
   this.height = height;
  }
  public boolean contains(float x, float y){
    return x >= this.x && x <= this.x + width && y >= this.y && y <= this.y + height;
  }
  public void draw(){
    fill(c);
    ellipse(x, y, width, height);
    noFill();
    stroke(0);
    ellipse(this.x, this.y, this.width, this.height);
  }
}
public static float distance(Point p1, Point p2) {
  return pow(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5f);
}
public static float distance(float x1, float y1, float x2, float y2) {
  return pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5f);
}
public static float nChoosek(int n, int k) {
  return (float)fact(n) / (float)(fact(k) * fact(n - k));
}
public static final int fact(int num) {
  if (num == 0) return 1;
  return num == 1 ? 1 : fact(num - 1)*num;
}
public static Matrix ElementwiseMultiplication(Matrix m1, Matrix m2) {
  Matrix m = new Matrix(m1.getRowDimension(), m1.getColumnDimension());
  for (int i = 0; i < m.getRowDimension(); i++) {
    for (int j = 0; j < m.getColumnDimension(); j++) {
      m.set(i, j, m1.get(i, j) * m2.get(i, j));
    }
  }
  return m;
}
public static boolean isInteger(String s) {
  boolean isValidInteger = false;
  try
  {
    Integer.parseInt(s);

    // s is a valid integer

    isValidInteger = true;
  }
  catch (NumberFormatException ex)
  {
    // s is not an integer
  }

  return isValidInteger;
}
public static void printMatrix(Matrix m) {
  for (int i = 0; i < m.getRowDimension(); i++) {
    for (int j = 0; j < m.getColumnDimension(); j++) {
      print(m.get(i, j) + " ");
    }
    println();
  }
  println();
  println();
}
  public void settings() {  size(900, 500);  noSmooth(); }
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "NeuralNets" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
